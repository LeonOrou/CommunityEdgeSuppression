import os
import torch
import numpy as np
import wandb
import yaml
import argparse
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN, ItemKNN, MultiVAE
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


def parse_arguments():
    """Parse command line arguments for standard model hyperparameters."""
    parser = argparse.ArgumentParser()

    # Basic settings
    parser.add_argument("--model", type=str, default="LightGCN", choices=["LightGCN", "ItemKNN", "MultiVAE"])
    parser.add_argument("--dataset", type=str, default="ml-100k")
    parser.add_argument("--seed", type=int, default=42)

    # Common hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embedding_size", type=int, default=64)
    # parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Model-specific hyperparameters
    # LightGCN
    parser.add_argument("--n_layers", type=int, default=3)

    # ItemKNN
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--shrink", type=float, default=0.0)

    # MultiVAE
    parser.add_argument("--hidden_dimension", type=int, default=600)
    parser.add_argument("--latent_dimension", type=int, default=200)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--anneal_cap", type=float, default=0.2)
    parser.add_argument("--total_anneal_steps", type=int, default=200000)

    # Evaluation
    parser.add_argument("--topk", type=int, nargs='+', default=[10, 20, 50])
    parser.add_argument("--valid_metric", type=str, default="ndcg@10")

    # Other settings
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--wandb_project", type=str, default="RecBole_StandardHyperparamTuning")

    # Learning rate scheduler arguments
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "multistep", "exponential", "cosine", "plateau"])
    parser.add_argument("--step_size", type=int, default=10, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="Learning rate decay factor")
    parser.add_argument("--milestones", type=int, nargs='+', default=[30, 60, 90], help="MultiStepLR milestones")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR for cosine or plateau scheduler")
    parser.add_argument("--patience", type=int, default=5, help="ReduceLROnPlateau patience")
    parser.add_argument("--t_max", type=int, default=50, help="CosineAnnealingLR T_max")

    return parser.parse_args()


def setup_device(gpu):
    """Setup device (CPU/GPU)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def setup_config(args, device, seed=42):
    """Setup RecBole configuration with hyperparameters."""
    # Common parameters

    config_dict = {
        'seed': args.seed,
        'reproducibility': True,
        'device': device,
        'learning_rate': args.learning_rate,
        'embedding_size': args.embedding_size,
    }

    # Add model-specific parameters
    if args.model == 'LightGCN':
        config_dict.update({
            'n_layers': args.n_layers,
            'train_batch_size': 512,
            'eval_batch_size': 512,
            'epoch': 200,
        })
    elif args.model == 'ItemKNN':
        config_dict.update({
            'k': args.k,
            'shrink': args.shrink,
        })
    elif args.model == 'MultiVAE':
        config_dict.update({
            'hidden_dimension': args.hidden_dimension,
            'latent_dimension': args.latent_dimension,
            'dropout_prob': args.dropout_prob,
            'anneal_cap': args.anneal_cap,
            'total_anneal_steps': args.total_anneal_steps,
            'train_batch_size': 512,
            'eval_batch_size': 512,
            'epoch': 200,
        })

    # Add scheduler configuration if enabled
    if args.scheduler != "none":
        config_dict.update({
            'scheduler': args.scheduler,
            'scheduler_gamma': args.gamma,
        })
        if args.scheduler == "step":
            config_dict["step_size"] = args.step_size
        elif args.scheduler == "multistep":
            config_dict["milestones"] = args.milestones
        elif args.scheduler == "cosine":
            config_dict["t_max"] = args.t_max
            config_dict["min_lr"] = args.min_lr
        elif args.scheduler == "plateau":
            config_dict["patience"] = args.patience
            config_dict["min_lr"] = args.min_lr

    # Create config object
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[f'{args.dataset}_config.yaml'],
        config_dict=config_dict
    )
    config['device'] = device

    return config


def prepare_dataset(config):
    """Create and prepare dataset."""
    dataset = create_dataset(config)
    logger = getLogger()
    logger.info(dataset)
    return data_preparation(config, dataset)


def initialize_wandb(args, config):
    """Initialize Weights & Biases for experiment tracking."""
    # Extract relevant config parameters for wandb logging
    wandb_config = {
        "model": args.model,
        "dataset": args.dataset,
        "learning_rate": args.learning_rate,
        "embedding_size": args.embedding_size,
        "seed": args.seed,
        "valid_metric": args.valid_metric
    }

    # Add model-specific parameters
    if args.model == 'LightGCN':
        wandb_config.update({"n_layers": args.n_layers})
    elif args.model == 'ItemKNN':
        wandb_config.update({"k": args.k, "shrink": args.shrink})
    elif args.model == 'MultiVAE':
        wandb_config.update({
            "hidden_dimension": args.hidden_dimension,
            "latent_dimension": args.latent_dimension,
            "dropout_prob": args.dropout_prob,
            "anneal_cap": args.anneal_cap,
            "total_anneal_steps": args.total_anneal_steps
        })

    # Add scheduler info for WandB logging if enabled
    if args.scheduler != "none":
        wandb_config.update({
            "scheduler": args.scheduler,
            "scheduler_gamma": args.gamma
        })
        if args.scheduler == "step":
            wandb_config.update({"step_size": args.step_size})
        elif args.scheduler == "multistep":
            wandb_config.update({"milestones": args.milestones})
        elif args.scheduler == "cosine":
            wandb_config.update({"t_max": args.t_max, "min_lr": args.min_lr})
        elif args.scheduler == "plateau":
            wandb_config.update({"patience": args.patience, "min_lr": args.min_lr})

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.model}_{args.dataset}_lr{args.learning_rate}_{args.scheduler}",
        config=wandb_config
    )

    return run


def initialize_model(model_name, config, train_data):
    """Initialize the recommendation model."""
    if model_name == 'LightGCN':
        model = LightGCN(config, train_data.dataset).to(config['device'])
    elif model_name == 'ItemKNN':
        model = ItemKNN(config, train_data.dataset).to(config['device'])
    elif model_name == 'MultiVAE':
        model = MultiVAE(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError(f"Model {model_name} not supported")

    logger = getLogger()
    logger.info(model)
    return model


def create_scheduler(args, optimizer):
    """Create learning rate scheduler based on command line arguments."""
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.min_lr)
    elif args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma,
                                                          patience=args.patience, min_lr=args.min_lr)
    else:  # "none"
        return None


def train_and_evaluate(config, model, train_data, valid_data, test_data):
    """Train and evaluate the model."""
    trainer = Trainer(config, model)

    # Train the model
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=True, saved=True
    )

    # Test the model
    test_result = trainer.evaluate(test_data)

    return best_valid_score, best_valid_result, test_result


def log_results_to_wandb(best_valid_score, best_valid_result, test_result):
    """Log training results to wandb."""
    # Log validation results
    wandb.log({"best_valid_score": best_valid_score})
    for metric, value in best_valid_result.items():
        wandb.log({f"valid_{metric}": value})

    # Log test results
    for metric, value in test_result.items():
        wandb.log({f"test_{metric}": value})


def load_config_from_yaml(dataset_name):
    """Load configuration from YAML file."""
    with open(f'{dataset_name}_config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
        return {
            'rating_col_name': config_file['RATING_FIELD'],
            'topk': config_file['topk'],
        }


def main():
    # Parse arguments
    args = parse_arguments()

    # Set up device
    device = setup_device(args.gpu)

    # Set seed for reproducibility
    init_seed(args.seed, True)

    # Setup configuration
    config = setup_config(args, device)
    print(f'Running with config: {config.variable_config_dict}')

    # Initialize logger
    init_logger(config)
    logger = getLogger()

    # Initialize wandb
    wandb_run = initialize_wandb(args, config)

    # Prepare datasets
    train_data, valid_data, test_data = prepare_dataset(config)

    # Initialize model
    model = initialize_model(args.model, config, train_data)

    # Log scheduler usage if enabled
    if args.scheduler != "none":
        logger.info(f"Using {args.scheduler} learning rate scheduler")

    # Train and evaluate
    best_valid_score, best_valid_result, test_result = train_and_evaluate(
        config=config,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data
    )

    # Log results to wandb
    log_results_to_wandb(best_valid_score, best_valid_result, test_result)

    # Save model ID and finish wandb run
    model_id = np.random.randint(0, 100000)
    wandb.save(f"{args.model}_{args.dataset}_ID{model_id}.h5")
    wandb_run.finish()

    # Log final results
    logger.info(f"Best valid score: {best_valid_score}")
    logger.info(f"Best valid result: {best_valid_result}")
    logger.info(f"Test result: {test_result}")


if __name__ == "__main__":
    main()
