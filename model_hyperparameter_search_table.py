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

    for line_num, line in enumerate(lines):
        try:
            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            # Parse the JSON directly - each line should be valid JSON
            data = json.loads(line)

            # Extract hyperparameters from experiment_config (first line)
            if data.get('event') == 'experiment_config':
                hyperparams = {
                    'model_name': data.get('model_name'),
                    'shrink': data.get('shrink'),
                    'item_knn_topk': data.get('item_knn_topk'),
                    'hidden_dimension': data.get('hidden_dimension'),
                    'latent_dimension': data.get('latent_dimension'),
                    'anneal_cap': data.get('anneal_cap'),
                    'embedding_dim': data.get('embedding_dim'),
                    'n_layers': data.get('n_layers')
                }

            # Extract cv_avg NDCG@10 (last metrics_report entry)
            elif data.get('event') == 'metrics_report' and data.get('stage') == 'cv_avg':
                cv_avg_ndcg = data['metrics']['top@10']['ndcg']

        except json.JSONDecodeError as e:
            # Skip lines that are not valid JSON (like the dataset statistics line)
            if line_num == 1:  # Second line is often dataset statistics
                continue
            print(f"JSON decode error in {filepath} at line {line_num + 1}: {e}")
            continue
        except Exception as e:
            print(f"Error parsing line {line_num + 1} in {filepath}: {e}")
            continue

    return hyperparams, cv_avg_ndcg


def find_optimal_hyperparameters(dataset: str = "ml-100k") -> pd.DataFrame:
    """
    Find optimal hyperparameters for each model based on highest NDCG@10.

    Args:
        dataset: Dataset name (default: "ml-100k")

    Returns:
        DataFrame with optimal hyperparameters for each model
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
    log_pattern = f"logs/{dataset}_*_????_??????.log"
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
                relevant_params['log_file'] = log_file
                model_results[model_name].append(relevant_params)

    # Find optimal hyperparameters for each model
    optimal_results = []

    for model, results in model_results.items():
        if results:
            # Sort by NDCG@10 in descending order and get the best
            best_result = max(results, key=lambda x: x['ndcg@10'])

            # Create row for the table
            row = {'Model': model}
            row.update({k: v for k, v in best_result.items() if k not in ['log_file', 'ndcg@10']})
            row['Best NDCG@10'] = best_result['ndcg@10']
            row['Log File'] = best_result['log_file'].split('/')[-1]  # Just filename

            optimal_results.append(row)

    # Create DataFrame
    df = pd.DataFrame(optimal_results)

    # Reorder columns for better readability
    column_order = ['Model']
    for model, params in model_hyperparams.items():
        column_order.extend(params)
    column_order.extend(['Best NDCG@10', 'Log File'])

    # Only keep columns that exist in the DataFrame
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    return df


def main():
    """
    Main function to extract and display optimal hyperparameters.
    """
    # Extract optimal hyperparameters
    optimal_df = find_optimal_hyperparameters("ml-100k")

    # Display results
    print("\nOptimal Hyperparameters for ml-100k Dataset (based on NDCG@10):")
    print("=" * 100)
    print(optimal_df.to_string(index=False))

    # Save to CSV
    output_file = "optimal_hyperparameters_ml-100k.csv"
    optimal_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Display summary statistics
    print("\nSummary Statistics:")
    for _, row in optimal_df.iterrows():
        model = row['Model']
        ndcg = row['Best NDCG@10']
        print(f"- {model}: Best NDCG@10 = {ndcg:.6f}")

        # Display hyperparameters
        if model == 'ItemKNN':
            print(f"  Hyperparameters: shrink={row.get('shrink')}, item_knn_topk={row.get('item_knn_topk')}")
        elif model == 'MultiVAE':
            print(f"  Hyperparameters: hidden_dimension={row.get('hidden_dimension')}, "
                  f"latent_dimension={row.get('latent_dimension')}, anneal_cap={row.get('anneal_cap')}")
        elif model == 'LightGCN':
            print(f"  Hyperparameters: embedding_dim={row.get('embedding_dim')}, n_layers={row.get('n_layers')}")


if __name__ == "__main__":
    main()