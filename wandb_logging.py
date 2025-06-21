import wandb
import os


def init_wandb(config, offline=False):
    # Initialize wandb with comprehensive config logging
        wandb_config = {
            # Model configuration
            'model_name': config.model_name,
            'dataset_name': config.dataset_name,
            'embedding_dim': getattr(config, 'embedding_dim', None),
            'n_layers': getattr(config, 'n_layers', None),
            'batch_size': getattr(config, 'batch_size', None),
            'epochs': getattr(config, 'epochs', None),
            'learning_rate': getattr(config, 'learning_rate', None),
            'hidden_dimension': getattr(config, 'hidden_dimension', None),
            'latent_dimension': getattr(config, 'latent_dimension', None),
            'reg': config.reg if hasattr(config, 'reg') else None,
            'item_knn_topk': getattr(config, 'item_knn_topk', None),
            'shrink': getattr(config, 'shrink', None),

            # Community suppression parameters
            'use_suppression': config.use_suppression,
            'community_suppression': config.community_suppression,
            'users_top_percent': config.users_top_percent,
            'items_top_percent': config.items_top_percent,
            'users_dec_perc_suppr': config.users_dec_perc_suppr,
            'items_dec_perc_suppr': config.items_dec_perc_suppr,
            'suppress_power_nodes_first': config.suppress_power_nodes_first,

            # Evaluation parameters
            'evaluate_top_k': getattr(config, 'evaluate_top_k', [10, 20, 50, 100]),
            'patience': getattr(config, 'patience', 10),

            # System info
            'device': str(config.device),
            'seed': 42,
        }

        # Initialize wandb run
        if offline:
            os.environ["WANDB_MODE"] = "offline"
        else:
            wandb.login(key="d234bc98a4761bff39de0e5170df00094ac42269")
        wandb.init(
            project="CommunitySuppression",
            config=wandb_config,
            name=f"{config.model_name}_{config.dataset_name}_dropout_{config.use_suppression}",
            tags=[config.model_name, config.dataset_name, "cross-validation"]
        )


def log_metrics_to_wandb(validation_metrics, config, stage):

    log_dict = {}

    for k in config.evaluate_top_k:
        # Accuracy metrics
        log_dict[f'{stage}/ndcg@{k}'] = validation_metrics[k]['ndcg']
        log_dict[f'{stage}/recall@{k}'] = validation_metrics[k]['recall']
        log_dict[f'{stage}/precision@{k}'] = validation_metrics[k]['precision']
        log_dict[f'{stage}/mrr@{k}'] = validation_metrics[k]['mrr']
        log_dict[f'{stage}/hit_rate@{k}'] = validation_metrics[k]['hit_rate']
        log_dict[f'{stage}/item_coverage@{k}'] = validation_metrics[k]['item_coverage']
        log_dict[f'{stage}/gini_index@{k}'] = validation_metrics[k]['gini_index']
        log_dict[f'{stage}/simpson_index_genre@{k}'] = validation_metrics[k]['simpson_index_genre']
        log_dict[f'{stage}/intra_list_diversity@{k}'] = validation_metrics[k]['intra_list_diversity']
        log_dict[f'{stage}/normalized_genre_entropy@{k}'] = validation_metrics[k]['normalized_genre_entropy']
        log_dict[f'{stage}/unique_genres_count@{k}'] = validation_metrics[k]['unique_genres_count']
        log_dict[f'{stage}/popularity_lift@{k}'] = validation_metrics[k]['popularity_lift']
        log_dict[f'{stage}/popularity_calibration@{k}'] = validation_metrics[k]['popularity_calibration']

    wandb.log(log_dict)


