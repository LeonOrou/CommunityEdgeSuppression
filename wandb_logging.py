import wandb


def init_wandb(config):
    # Initialize wandb with comprehensive config logging
        wandb_config = {
            # Model configuration
            'model_name': config.model_name,
            'dataset_name': config.dataset_name,
            'embedding_dim': getattr(config, 'embedding_dim', None),
            'n_layers':getattr(config, 'n_layers', None),
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': getattr(config, 'learning_rate', None),
            'hidden_dimension': getattr(config, 'hidden_dimension', None),
            'latent_dimension': getattr(config, 'latent_dimension', None),
            'reg': config.reg if hasattr(config, 'reg') else None,
            'item_knn_topk': getattr(config, 'item_knn_topk', None),
            'shrink': getattr(config, 'shrink', None),

            # Community suppression parameters
            'use_dropout': config.use_dropout,
            'community_suppression': config.community_suppression,
            'users_top_percent': config.users_top_percent,
            'items_top_percent': config.items_top_percent,
            'users_dec_perc_drop': config.users_dec_perc_drop,
            'items_dec_perc_drop': config.items_dec_perc_drop,
            'drop_only_power_nodes': config.drop_only_power_nodes,

            # Evaluation parameters
            'evaluate_top_k': getattr(config, 'evaluate_top_k', [10, 20, 50, 100]),
            'patience': getattr(config, 'patience', 10),

            # System info
            'device': str(config.device),
            'seed': 42,
        }

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
    log_dict = {}

    for k in config.evaluate_top_k:
        log_dict[f'fold_{fold_num}/val_ndcg@{k}'] = fold_results[k][f'ndcg']
        log_dict[f'fold_{fold_num}/val_recall@{k}'] = fold_results[k][f'recall']
        log_dict[f'fold_{fold_num}/val_precision@{k}'] = fold_results[k][f'precision']
        log_dict[f'fold_{fold_num}/val_mrr@{k}'] = fold_results[k][f'mrr']
        log_dict[f'fold_{fold_num}/val_hit_rate@{k}'] = fold_results[k][f'hit_rate']
        log_dict[f'fold_{fold_num}/val_item_coverage@{k}'] = fold_results[k][f'item_coverage']
        log_dict[f'fold_{fold_num}/val_gini_index@{k}'] = fold_results[k][f'gini_index']
        log_dict[f'fold_{fold_num}/val_simpson_index_genre@{k}'] = fold_results[k][f'simpson_index_genre']
        log_dict[f'fold_{fold_num}/val_intra_list_diversity@{k}'] = fold_results[k][f'intra_list_diversity']
        log_dict[f'fold_{fold_num}/val_normalized_genre_entropy@{k}'] = fold_results[k][f'normalized_genre_entropy']
        log_dict[f'fold_{fold_num}/val_unique_genres_count@{k}'] = fold_results[k][f'unique_genres_count']
        log_dict[f'fold_{fold_num}/val_popularity_lift@{k}'] = fold_results[k][f'popularity_lift']
        log_dict[f'fold_{fold_num}/val_popularity_calibration@{k}'] = fold_results[k][f'popularity_calibration']
        log_dict[f'fold_{fold_num}/val_user_community_bias@{k}'] = fold_results[k][f'user_community_bias']

    wandb.log(log_dict)


def log_cv_summary_to_wandb(cv_summary, config):
    """Log cross-validation summary to wandb"""

    log_dict = {}

    for k in config.evaluate_top_k:
        # Accuracy metrics
        log_dict[f'cv_avg/ndcg@{k}'] = cv_summary[k]['ndcg']
        log_dict[f'cv_avg/recall@{k}'] = cv_summary[k]['recall']
        log_dict[f'cv_avg/precision@{k}'] = cv_summary[k]['precision']
        log_dict[f'cv_avg/mrr@{k}'] = cv_summary[k]['mrr']
        log_dict[f'cv_avg/hit_rate@{k}'] = cv_summary[k]['hit_rate']
        log_dict[f'cv_avg/item_coverage@{k}'] = cv_summary[k]['item_coverage']
        log_dict[f'cv_avg/gini_index@{k}'] = cv_summary[k]['gini_index']
        log_dict[f'cv_avg/simpson_index_genre@{k}'] = cv_summary[k]['simpson_index_genre']
        log_dict[f'cv_avg/intra_list_diversity@{k}'] = cv_summary[k]['intra_list_diversity']
        log_dict[f'cv_avg/normalized_genre_entropy@{k}'] = cv_summary[k]['genre_entropy']
        log_dict[f'cv_avg/unique_genres_count@{k}'] = cv_summary[k]['unique_genres_count']
        log_dict[f'cv_avg/popularity_lift@{k}'] = cv_summary[k]['popularity_lift']
        log_dict[f'cv_avg/popularity_calibration@{k}'] = cv_summary[k]['popularity_calibration']
        log_dict[f'cv_avg/user_community_bias@{k}'] = cv_summary[k]['user_community_bias']

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
        log_dict[f'test/item_coverage@{k}'] = test_metrics[k]['item_coverage']
        log_dict[f'test/gini_index@{k}'] = test_metrics[k]['gini_index']
        log_dict[f'test/simpson_index_genre@{k}'] = test_metrics[k]['simpson_index_genre']
        log_dict[f'test/intra_list_diversity@{k}'] = test_metrics[k]['intra_list_diversity']
        log_dict[f'test/normalized_genre_entropy@{k}'] = test_metrics[k]['normalized_genre_entropy']
        log_dict[f'test/unique_genres_count@{k}'] = test_metrics[k]['unique_genres_count']
        log_dict[f'test/popularity_lift@{k}'] = test_metrics[k]['popularity_lift']
        log_dict[f'test/popularity_calibration@{k}'] = test_metrics[k]['popularity_calibration']

    wandb.log(log_dict)


def _log_multivae_training_metrics(epoch, epoch_loss, val_ndcg, current_lr, kl_weight,
                                   patience_counter, fold_num):
    log_dict = {
        'epoch': epoch,
        'train_loss': epoch_loss,
        'val_ndcg@10': val_ndcg,
        'learning_rate': current_lr,
        'kl_annealing_weight': kl_weight,
        'patience_counter': patience_counter
    }

    if fold_num is not None:
        log_dict = {f'fold_{fold_num}/{k}': v for k, v in log_dict.items()}
    else:
        log_dict = {f'final_training/{k}': v for k, v in log_dict.items()}

    wandb.log(log_dict)


def _log_multivae_final_results(best_epoch, best_val_ndcg, total_epochs, early_stopped, fold_num):
    """Log final MultiVAE training results to WandB."""
    final_log_dict = {
        'best_epoch': best_epoch,
        'best_val_ndcg@10': best_val_ndcg,
        'total_epochs': total_epochs,
        'early_stopped': early_stopped
    }

    if fold_num is not None:
        final_log_dict = {f'fold_{fold_num}/final_{k}': v for k, v in final_log_dict.items()}
    else:
        final_log_dict = {f'final_training/final_{k}': v for k, v in final_log_dict.items()}

    wandb.log(final_log_dict)


def _log_training_metrics(epoch, epoch_loss, val_ndcg, current_lr, patience_counter,
                          config, fold_num):
    """Log training metrics to WandB."""
    log_dict = {
        'epoch': epoch,
        'train_loss': epoch_loss,
        'val_ndcg@10': val_ndcg,
        'learning_rate': current_lr,
        'patience_counter': patience_counter,
        'suppression_enabled': config.use_dropout
    }

    # Add fold-specific prefix if in cross-validation
    if fold_num is not None:
        log_dict = {f'fold_{fold_num}/{k}': v for k, v in log_dict.items()}
    else:
        log_dict = {f'final_training/{k}': v for k, v in log_dict.items()}

    wandb.log(log_dict)


def _log_final_training_results(best_epoch, best_val_ndcg, total_epochs, early_stopped, fold_num):
    """Log final training results to WandB."""
    final_log_dict = {
        'best_epoch': best_epoch,
        'best_val_ndcg@10': best_val_ndcg,
        'total_epochs': total_epochs,
        'early_stopped': early_stopped
    }

    if fold_num is not None:
        final_log_dict = {f'fold_{fold_num}/final_{k}': v for k, v in final_log_dict.items()}
    else:
        final_log_dict = {f'final_training/final_{k}': v for k, v in final_log_dict.items()}

    wandb.log(final_log_dict)


def _log_itemknn_training_start(model, num_interactions, fold_num, verbose):
    """Log ItemKNN training start information."""
    log_dict = {
        'training_start': True,
        'num_neighbors': model.k,
        'shrinkage': model.shrink,
        'bm25_k1': model.bm25_k1,
        'bm25_b': model.bm25_b,
        'num_training_interactions': num_interactions,
        'num_users': model.num_users,
        'num_items': model.num_items
    }

    # Add fold-specific prefix if in cross-validation
    if fold_num is not None:
        log_dict = {f'fold_{fold_num}/itemknn_{k}': v for k, v in log_dict.items()}
    else:
        log_dict = {f'final_training/itemknn_{k}': v for k, v in log_dict.items()}

    wandb.log(log_dict)

    if verbose:
        print(f"  Building item similarity matrix...")
        print(f"  Parameters: k={model.k}, shrink={model.shrink}, "
              f"BM25(k1={model.bm25_k1}, b={model.bm25_b})")


