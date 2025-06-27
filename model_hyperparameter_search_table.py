import json
import glob
import pandas as pd
from typing import Dict, List, Tuple


def parse_log_file(filepath: str) -> Tuple[Dict, float]:
    """
    Parse a log file and extract hyperparameters and cv_avg NDCG@10.

    Args:
        filepath: Path to the log file

    Returns:
        Tuple of (hyperparameters dict, ndcg@10 value)
    """
    hyperparams = {}
    cv_avg_ndcg = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]

    if not lines:
        print(f"Warning: Empty file {filepath}")
        return hyperparams, cv_avg_ndcg

    # Parse first line for experiment config
    try:
        first_line_data = json.loads(lines[0])
        if first_line_data.get('event') == 'experiment_config':
            hyperparams = {
                'model_name': first_line_data.get('model_name'),
                'shrink': first_line_data.get('shrink'),
                'item_knn_topk': first_line_data.get('item_knn_topk'),
                'hidden_dimension': first_line_data.get('hidden_dimension'),
                'latent_dimension': first_line_data.get('latent_dimension'),
                'anneal_cap': first_line_data.get('anneal_cap'),
                'embedding_dim': first_line_data.get('embedding_dim'),
                'n_layers': first_line_data.get('n_layers')
            }
    except json.JSONDecodeError as e:
        print(f"Error parsing first line in {filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error parsing first line in {filepath}: {e}")

    # Parse last line for cv_avg metrics
    try:
        last_line_data = json.loads(lines[-1])
        if (last_line_data.get('event') == 'metrics_report' and
                last_line_data.get('stage') == 'cv_avg'):
            cv_avg_ndcg = last_line_data['metrics']['top@10']['ndcg']
        else:
            print(f"Warning: Last line in {filepath} is not cv_avg metrics")
    except json.JSONDecodeError as e:
        print(f"Error parsing last line in {filepath}: {e}")
    except KeyError as e:
        print(f"Missing expected key in {filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error parsing last line in {filepath}: {e}")

    return hyperparams, cv_avg_ndcg


def find_top_hyperparameters(dataset: str = "ml-100k", top_k: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Find top K hyperparameter configurations for each model based on highest NDCG@10.

    Args:
        dataset: Dataset name (default: "ml-100k")
        top_k: Number of top configurations to return per model (default: 10)

    Returns:
        Dictionary with model names as keys and DataFrames of top configurations as values
    """
    # Define models and their specific hyperparameters
    model_hyperparams = {
        'ItemKNN': ['shrink', 'item_knn_topk'],
        'MultiVAE': ['hidden_dimension', 'latent_dimension', 'anneal_cap'],
        'LightGCN': ['embedding_dim', 'n_layers']
    }

    # Store results for each model
    model_results = {model: [] for model in model_hyperparams}

    # Find all log files matching the pattern
    log_pattern = f"logs/{dataset} model hyperparameter search/{dataset}_*_????_??????.log"
    log_files = glob.glob(log_pattern)

    print(f"Found {len(log_files)} log files for dataset {dataset}")

    # Process each log file
    for log_file in log_files:
        hyperparams, ndcg = parse_log_file(log_file)

        if hyperparams.get('model_name') and ndcg is not None:
            model_name = hyperparams['model_name']
            if model_name in model_results:
                # Extract only relevant hyperparameters for this model
                relevant_params = {
                    param: hyperparams.get(param)
                    for param in model_hyperparams[model_name]
                    if hyperparams.get(param) is not None
                }
                relevant_params['ndcg@10'] = ndcg
                relevant_params['log_file'] = log_file.split('/')[-1]  # Just filename
                model_results[model_name].append(relevant_params)

    # Create DataFrames with top K results for each model
    top_results = {}

    for model, results in model_results.items():
        if results:
            # Sort by NDCG@10 in descending order and get top K
            sorted_results = sorted(results, key=lambda x: x['ndcg@10'], reverse=True)[:top_k]

            # Create DataFrame for this model
            df_data = []
            for rank, result in enumerate(sorted_results, 1):
                row = {'Rank': rank}
                # Add hyperparameters in the order they're defined
                for param in model_hyperparams[model]:
                    if param in result:
                        row[param] = result[param]
                row['NDCG@10'] = result['ndcg@10']
                row['Log File'] = result['log_file']
                df_data.append(row)

            df = pd.DataFrame(df_data)
            top_results[model] = df

    return top_results


def create_combined_table(top_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a combined table with all models' top results.

    Args:
        top_results: Dictionary with model names as keys and DataFrames as values

    Returns:
        Combined DataFrame with all results
    """
    combined_data = []

    for model, df in top_results.items():
        if not df.empty:
            # Add model column to each row
            df_copy = df.copy()
            df_copy.insert(0, 'Model', model)
            combined_data.append(df_copy)

    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    else:
        return pd.DataFrame()


def display_results(dataset: str = "ml-100k") -> None:
    """
    Main function to extract and display top hyperparameters for all models.
    """
    # Extract top hyperparameters
    top_results = find_top_hyperparameters(dataset=dataset, top_k=10)

    # Display results for each model separately
    print("\nTop 10 Hyperparameter Configurations for ml-100k Dataset (ranked by NDCG@10):")
    print("=" * 120)

    for model, df in top_results.items():
        if not df.empty:
            print(f"\n{model}:")
            print("-" * 100)
            print(df.to_string(index=False))

            # Save individual model results
            output_file = f"top10_hyperparameters_{model}_ml-100k.csv"
            df.to_csv(output_file, index=False)
            print(f"\n{model} results saved to: {output_file}")

    # Create and save combined table
    combined_df = create_combined_table(top_results)
    if not combined_df.empty:
        combined_output = "top10_hyperparameters_all_models_ml-100k.csv"
        combined_df.to_csv(combined_output, index=False)
        print(f"\nCombined results saved to: {combined_output}")

        # Display summary statistics
        print("\n" + "=" * 120)
        print("Summary - Best Configuration for Each Model:")
        print("-" * 80)

        for model in ['ItemKNN', 'MultiVAE', 'LightGCN']:
            model_data = combined_df[combined_df['Model'] == model]
            if not model_data.empty:
                best_row = model_data.iloc[0]
                print(f"\n{model}:")
                print(f"  Best NDCG@10: {best_row['NDCG@10']:.6f}")

                if model == 'ItemKNN':
                    print(f"  Hyperparameters: shrink={best_row['shrink']}, item_knn_topk={best_row['item_knn_topk']}")
                elif model == 'MultiVAE':
                    params_str = f"  Hyperparameters: hidden_dimension={best_row['hidden_dimension']}, "
                    params_str += f"latent_dimension={best_row['latent_dimension']}"
                    if pd.notna(best_row.get('anneal_cap')):
                        params_str += f", anneal_cap={best_row['anneal_cap']}"
                    print(params_str)
                elif model == 'LightGCN':
                    print(
                        f"  Hyperparameters: embedding_dim={best_row['embedding_dim']}, n_layers={best_row['n_layers']}")


display_results(dataset='ml-1m')

