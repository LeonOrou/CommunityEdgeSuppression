import wandb


def init_wandb(config):
    # Initialize wandb with comprehensive config logging
        wandb_config = {
            # Model configuration
            'model_name': config.model_name,
            'dataset_name': config.dataset_name,
            'embedding_dim': config.embedding_dim,
            'n_layers': config.n_layers,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'reg': config.reg,

            # Community suppression parameters
            'use_dropout': config.use_dropout,
            'community_suppression': config.community_suppression,
            'users_top_percent': config.users_top_percent,
            'items_top_percent': config.items_top_percent,
            'users_dec_perc_drop': config.users_dec_perc_drop,
            'items_dec_perc_drop': config.items_dec_perc_drop,
            'drop_only_power_nodes': config.drop_only_power_nodes,

            # Evaluation parameters
            'evaluate_top_k': config.evaluate_top_k,
            'patience': getattr(config, 'patience', 50),

            # System info
            'device': str(config.device),
            'seed': 42
        }

        # Add model-specific parameters
        if hasattr(config, 'item_knn_topk'):
            wandb_config['item_knn_topk'] = config.item_knn_topk
        if hasattr(config, 'shrink'):
            wandb_config['shrink'] = config.shrink

        # Initialize wandb run
        wandb.login(key="d234bc98a4761bff39de0e5170df00094ac42269")
        wandb.init(
            project="CommunitySuppression",
            config=wandb_config,
            name=f"{config.model_name}_{config.dataset_name}_dropout_{config.use_dropout}",
            tags=[config.model_name, config.dataset_name, "cross-validation"]
        )


def log_fold_metrics_to_wandb(fold_num, fold_results, config):
    """Log comprehensive fold metrics to wandb"""

    # Prepare metrics for logging
    log_dict = {}

    # Log all metrics for each k value
    for k in config.evaluate_top_k:
        # Accuracy metrics
        log_dict[f'fold_{fold_num}/val_ndcg@{k}'] = fold_results[f'val_ndcg@{k}']
        log_dict[f'fold_{fold_num}/val_recall@{k}'] = fold_results[f'val_recall@{k}']
        log_dict[f'fold_{fold_num}/val_precision@{k}'] = fold_results[f'val_precision@{k}']
        log_dict[f'fold_{fold_num}/val_mrr@{k}'] = fold_results[f'val_mrr@{k}']
        log_dict[f'fold_{fold_num}/val_hit_rate@{k}'] = fold_results[f'val_hit_rate@{k}']

        # Coverage and distribution metrics
        log_dict[f'fold_{fold_num}/val_item_coverage@{k}'] = fold_results[f'val_item_coverage@{k}']
        log_dict[f'fold_{fold_num}/val_gini_index@{k}'] = fold_results[f'val_gini_index@{k}']
        log_dict[f'fold_{fold_num}/val_simpson_index@{k}'] = fold_results[f'val_simpson_index@{k}']

        # Diversity metrics
        log_dict[f'fold_{fold_num}/val_simpson_index_genre@{k}'] = fold_results[f'val_simpson_index_genre@{k}']
        log_dict[f'fold_{fold_num}/val_intra_list_diversity@{k}'] = fold_results[f'val_intra_list_diversity@{k}']
        log_dict[f'fold_{fold_num}/val_normalized_genre_entropy@{k}'] = fold_results[
            f'val_normalized_genre_entropy@{k}']
        log_dict[f'fold_{fold_num}/val_unique_genres_count@{k}'] = fold_results[f'val_unique_genres_count@{k}']

        # Popularity metrics
        log_dict[f'fold_{fold_num}/val_popularity_lift@{k}'] = fold_results[f'val_popularity_lift@{k}']
        log_dict[f'fold_{fold_num}/val_popularity_calibration@{k}'] = fold_results[f'val_popularity_calibration@{k}']

    wandb.log(log_dict)


def log_cv_summary_to_wandb(cv_summary, config):
    """Log cross-validation summary to wandb"""

    log_dict = {}

    for k in config.evaluate_top_k:
        # Accuracy metrics
        log_dict[f'cv_avg/ndcg@{k}'] = cv_summary[k]['NDCG']
        log_dict[f'cv_avg/recall@{k}'] = cv_summary[k]['Recall']
        log_dict[f'cv_avg/precision@{k}'] = cv_summary[k]['Precision']
        log_dict[f'cv_avg/mrr@{k}'] = cv_summary[k]['MRR']
        log_dict[f'cv_avg/hit_rate@{k}'] = cv_summary[k]['Hit Rate']

        # Coverage and distribution metrics
        log_dict[f'cv_avg/item_coverage@{k}'] = cv_summary[k]['Item Coverage']
        log_dict[f'cv_avg/gini_index@{k}'] = cv_summary[k]['Gini Index']
        log_dict[f'cv_avg/simpson_index@{k}'] = cv_summary[k]['Simpson Index']

        # Diversity metrics
        log_dict[f'cv_avg/simpson_index_genre@{k}'] = cv_summary[k]['Simpson (Genre)']
        log_dict[f'cv_avg/intra_list_diversity@{k}'] = cv_summary[k]['Intra-list Diversity']
        log_dict[f'cv_avg/normalized_genre_entropy@{k}'] = cv_summary[k]['Genre Entropy']
        log_dict[f'cv_avg/unique_genres_count@{k}'] = cv_summary[k]['Unique Genres']

        # Popularity metrics
        log_dict[f'cv_avg/popularity_lift@{k}'] = cv_summary[k]['Popularity Lift']
        log_dict[f'cv_avg/popularity_calibration@{k}'] = cv_summary[k]['Pop. Calibration']

    wandb.log(log_dict)


def log_test_metrics_to_wandb(test_metrics, config):
    """Log final test metrics to wandb"""

    log_dict = {}

    for k in config.evaluate_top_k:
        # Accuracy metrics
        log_dict[f'test/ndcg@{k}'] = test_metrics[k]['ndcg']
        log_dict[f'test/recall@{k}'] = test_metrics[k]['recall']
        log_dict[f'test/precision@{k}'] = test_metrics[k]['precision']
        log_dict[f'test/mrr@{k}'] = test_metrics[k]['mrr']
        log_dict[f'test/hit_rate@{k}'] = test_metrics[k]['hit_rate']

        # Coverage and distribution metrics
        log_dict[f'test/item_coverage@{k}'] = test_metrics[k]['item_coverage']
        log_dict[f'test/gini_index@{k}'] = test_metrics[k]['gini_index']
        log_dict[f'test/simpson_index@{k}'] = test_metrics[k]['simpson_index']

        # Diversity metrics
        log_dict[f'test/simpson_index_genre@{k}'] = test_metrics[k]['simpson_index_genre']
        log_dict[f'test/intra_list_diversity@{k}'] = test_metrics[k]['intra_list_diversity']
        log_dict[f'test/normalized_genre_entropy@{k}'] = test_metrics[k]['normalized_genre_entropy']
        log_dict[f'test/unique_genres_count@{k}'] = test_metrics[k]['unique_genres_count']

        # Popularity metrics
        log_dict[f'test/popularity_lift@{k}'] = test_metrics[k]['popularity_lift']
        log_dict[f'test/popularity_calibration@{k}'] = test_metrics[k]['popularity_calibration']

    wandb.log(log_dict)


