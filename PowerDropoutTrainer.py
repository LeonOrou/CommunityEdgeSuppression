# custom recbole trainer which used power_node_edge_dropout before each episode

from recbole.trainer import Trainer
from time import time
from recbole.utils import (
    early_stopping,
    dict2str,
    set_color,
)
import torch
from utils_functions import power_node_edge_dropout
import os
import numpy as np
import copy
import pandas as pd
from recbole.data.interaction import Interaction
from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.sampler.sampler import RepeatableSampler


class PowerDropoutTrainer(Trainer):
    def __init__(self, config, model):
        super(PowerDropoutTrainer, self).__init__(config, model)
        ### adding learning rate scheduler ###
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                       patience=self.config.variable_config_dict['patience'],
                                                                       min_lr=self.config.variable_config_dict['min_lr'],)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()

            ### power_node_edge_dropout ###
            # saving the setting from train_data TrainDataLoader for later use
            # train_data_old = copy.deepcopy(train_data)

            train_data_coo = copy.deepcopy(train_data.dataset).inter_matrix()
            # combine row and col into torch.tensor of shape (n, 2) converting the data to numpy arrays and concatenating them
            indices = torch.tensor((train_data_coo.row, train_data_coo.col), dtype=torch.int32, device=self.device).T
            values = torch.unsqueeze(torch.tensor(train_data_coo.data, dtype=torch.int32, device=self.device), dim=0).T
            adj_tens = torch.cat((indices, values), dim=1).to(torch.int64)
            del train_data_coo, indices, values

            new_inter_feat = power_node_edge_dropout(adj_tens=adj_tens,
                                                     power_users_idx=self.config.variable_config_dict['power_users_ids'],
                                                     power_items_idx=self.config.variable_config_dict['power_items_ids'],
                                                     biased_user_edges_mask=self.config.variable_config_dict['biased_user_edges_mask'],
                                                     biased_item_edges_mask=self.config.variable_config_dict['biased_item_edges_mask'],
                                                     users_dec_perc_drop=self.config.variable_config_dict['users_dec_perc_drop'],
                                                     items_dec_perc_drop=self.config.variable_config_dict['items_dec_perc_drop'],
                                                     community_dropout_strength=self.config.variable_config_dict['community_dropout_strength'],
                                                     drop_only_power_nodes=self.config.variable_config_dict['drop_only_power_nodes'], )

            cpu_data = new_inter_feat.cpu().numpy()
            interaction_df = pd.DataFrame(cpu_data, columns=train_data.dataset.inter_feat.columns)
            # adding also all parameters to train_data_epoch that the original train_data had
            # make proper sampler and generator for the new data
            train_data_new = TrainDataLoader(config=self.config,
                   dataset=train_data.dataset.copy(new_inter_feat=Interaction(interaction_df)),
                   sampler=RepeatableSampler(phases='train',
                   dataset=train_data.dataset.copy(new_inter_feat=Interaction(interaction_df))),
                   shuffle=self.config['shuffle'])

            del cpu_data, interaction_df
            ### end power_node_edge_dropout ###

            train_loss = self._train_epoch(
                train_data_new, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )

                ### added lr-scheduler step ###
                self.lr_scheduler.step(valid_score)
                ###

                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    from recbole.trainer import Trainer
    import torch

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            # Add weights_only=False parameter to fix the loading issue
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
        return super().evaluate(eval_data, load_best_model=False, show_progress=show_progress)

