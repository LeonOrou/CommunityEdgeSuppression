import logging
import os
import random
import json
import datetime


def init_logging(config, log_level=logging.INFO):
    """
    Initialize Python logging and log the configuration in JSON format.

    Args:
        config: An object containing experiment configuration attributes.
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate a unique datatime ID for the experiment run
    datetime_id = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # Store the experiment_run_id in the config object for consistent use across log calls
    config.experiment_run_id = f"{config.dataset_name}_{config.model_name}_u{config.users_dec_perc_suppr}_s{config.community_suppression}_p{config.suppress_power_nodes_first}_{datetime_id}"
    log_filename = f"{config.experiment_run_id}.log"
    log_path = os.path.join(log_dir, log_filename)
    config.log_path = log_path  # Save log_path in config for reference

    # Configure logging to both file and console.
    # The format string here is simple, as the message content will be structured JSON.
    logging.basicConfig(
        level=log_level,
        format='%(message)s',  # Only log the message (the JSON), no timestamp/level
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(log_path, mode='a', encoding='utf-8')  # Log to file
        ]
    )

    # Prepare the initial configuration data as a dictionary
    config_data = {
        "event": "experiment_config",  # Type of log event
        "experiment_id": config.experiment_run_id,  # Unique ID for this run
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "embedding_dim": getattr(config, 'embedding_dim', None),
        "n_layers": getattr(config, 'n_layers', None),
        "batch_size": getattr(config, 'batch_size', None),
        "epochs": getattr(config, 'epochs', None),
        "learning_rate": getattr(config, 'learning_rate', None),
        "hidden_dimension": getattr(config, 'hidden_dimension', None),
        "latent_dimension": getattr(config, 'latent_dimension', None),
        "anneal_cap": getattr(config, 'anneal_cap', None),
        "reg": getattr(config, 'reg', None),
        "item_knn_topk": getattr(config, 'item_knn_topk', None),
        "shrink": getattr(config, 'shrink', None),
        "use_suppression": getattr(config, 'use_suppression', None),
        "community_suppression": getattr(config, 'community_suppression', None),
        "users_top_percent": getattr(config, 'users_top_percent', None),
        "items_top_percent": getattr(config, 'items_top_percent', None),
        "users_dec_perc_suppr": getattr(config, 'users_dec_perc_suppr', None),
        "items_dec_perc_suppr": getattr(config, 'items_dec_perc_suppr', None),
        "suppress_power_nodes_first": getattr(config, 'suppress_power_nodes_first', None),
        "evaluate_top_k": getattr(config, 'evaluate_top_k', [10, 20, 50, 100]),
        "patience": getattr(config, 'patience', 10),
        "device": str(getattr(config, 'device', 'cpu')),  # Ensure device is a string
        "seed": getattr(config, 'seed', 42),
    }
    # Log the configuration dictionary as a single-line JSON string (no newlines, no indent)
    logging.info(json.dumps(config_data, separators=(',', ':')))


def log_metrics(validation_metrics, config, stage):
    """
    Log metrics in JSON format, with each top-k group as a nested object.

    Args:
        validation_metrics: A dictionary containing metrics for different top-k values.
                            Expected structure: {k: {'ndcg': ..., 'recall': ..., ...}}
        config: An object containing experiment configuration, specifically 'experiment_run_id'
                and 'evaluate_top_k'.
        stage: A string indicating the stage of evaluation (e.g., "Validation_Epoch_1", "Test").
    """
    metrics_data = {}
    # Iterate through each k value specified in config.evaluate_top_k
    for k in getattr(config, 'evaluate_top_k', [10, 20, 50, 100]):
        # Check if metrics for the current k exist to avoid KeyError
        if k in validation_metrics:
            metrics_data[f"top@{k}"] = { # Key for each top-k metric group
                "ndcg": validation_metrics[k].get('ndcg'),
                "recall": validation_metrics[k].get('recall'),
                "precision": validation_metrics[k].get('precision'),
                "mrr": validation_metrics[k].get('mrr'),
                "hit_rate": validation_metrics[k].get('hit_rate'),
                "item_coverage": validation_metrics[k].get('item_coverage'),
                "gini_index": validation_metrics[k].get('gini_index'),
                "average_rec_popularity": validation_metrics[k].get('avg_rec_popularity'),
                "popularity_lift": validation_metrics[k].get('popularity_lift'),
                "pop_miscalibration": validation_metrics[k].get('pop_miscalibration'),
                "simpson_index_genre": validation_metrics[k].get('simpson_index_genre'),
                "intra_list_diversity": validation_metrics[k].get('intra_list_diversity'),
                "normalized_genre_entropy": validation_metrics[k].get('normalized_genre_entropy'),
                "unique_genres_count": validation_metrics[k].get('unique_genres_count'),
                "user_community_bias": validation_metrics[k].get('user_community_bias'),
            }
        else:
            logging.warning(f"Metrics for top@{k} not found in validation_metrics.")

    # Construct the full log entry as a dictionary
    log_entry = {
        "training_time": config.training_time,
        "event": "metrics_report",  # Type of log event
        "experiment_id": config.experiment_run_id,  # Consistent experiment ID
        "stage": stage,  # Stage of the experiment (e.g., validation, test)
        "metrics": metrics_data  # Nested metrics data
    }
    # Log the metrics dictionary as a single-line JSON string (no newlines, no indent)
    logging.info(json.dumps(log_entry, separators=(',', ':')))
