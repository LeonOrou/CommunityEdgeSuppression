import pandas as pd
import json
from pathlib import Path
from typing import List, Union, Dict, Any


def extract_experiment_results(
        csv_path: str = "logs/full_experiments_best_biases/experiment_results_best_biases_each_approach.csv",
        metrics: List[str] = ["ndcg", "user_community_bias", "recall", "precision", "hit_rate", "mrr", "item_coverage", "gini_index", "average_rec_popularity", "popularity_lift", "pop_miscalibration", "intra_list_diversity", "normalized_genre_entropy", "simpson_index_genre", "unique_genres_count"],
        topk: int = 10,
        output_format: str = "both"  # "table", "json", or "both"
) -> Dict[str, Any]:
    """
    Extract and format experiment results from CSV file.

    Args:
        csv_path: Path to the CSV file
        metrics: List of metrics to extract (e.g., ["ndcg", "recall", "precision"])
        topk: Top-K value for metrics (e.g., 10 for top@10)
        output_format: Output format - "table", "json", or "both"

    Returns:
        Dictionary containing formatted results
    """

    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        return {}
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}

    # Create metric column names for the specified topk
    metric_columns = [f"top@{topk}_{metric}" for metric in metrics]

    # Check if all requested metric columns exist
    missing_columns = [col for col in metric_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in CSV: {missing_columns}")
        metric_columns = [col for col in metric_columns if col in df.columns]

    if not metric_columns:
        print("Error: No valid metric columns found")
        return {}

    # Separate baseline and suppression experiments
    baseline_df = df[df['use_suppression'] == False].copy()
    suppression_df = df[df['use_suppression'] == True].copy()

    print(f"Found {len(baseline_df)} baseline experiments and {len(suppression_df)} suppression experiments")

    # Prepare the results structure
    results = {}

    # Process each dataset
    for dataset in df['dataset_name'].unique():
        results[dataset] = {}

        # Get baseline results for this dataset
        dataset_baseline = baseline_df[baseline_df['dataset_name'] == dataset]

        # Get suppression results for this dataset
        dataset_suppression = suppression_df[suppression_df['dataset_name'] == dataset]

        # Process each model in this dataset
        for model in df[df['dataset_name'] == dataset]['model_name'].unique():
            model_results = []

            # Add baseline row
            model_baseline = dataset_baseline[dataset_baseline['model_name'] == model]
            if not model_baseline.empty:
                baseline_row = {
                    'model_name': model,
                    'experiment_type': 'baseline',
                    'users_dec_perc_suppr': 'N/A',
                    'community_suppression': 'N/A',
                    'suppress_power_nodes_first': 'N/A'
                }
                # Add metric values
                for metric_col in metric_columns:
                    baseline_row[metric_col] = model_baseline[metric_col].iloc[0]

                model_results.append(baseline_row)

            # Add suppression experiment rows
            model_suppression = dataset_suppression[dataset_suppression['model_name'] == model]
            for _, row in model_suppression.iterrows():
                suppression_row = {
                    'model_name': model,
                    'experiment_type': 'suppression',
                    'users_dec_perc_suppr': row['users_dec_perc_suppr'],
                    'community_suppression': row['community_suppression'],
                    'suppress_power_nodes_first': row['suppress_power_nodes_first']
                }
                # Add metric values
                for metric_col in metric_columns:
                    suppression_row[metric_col] = row[metric_col]

                model_results.append(suppression_row)

            results[dataset][model] = model_results

    # Create formatted output
    formatted_results = {
        'parameters': {
            'metrics': metrics,
            'topk': topk,
            'csv_path': csv_path
        },
        'results': results
    }

    # Generate table format
    if output_format in ["table", "both"]:
        print("\n" + "=" * 100)
        print(f"EXPERIMENT RESULTS - Top@{topk} Metrics")
        print("=" * 100)

        for dataset_name, dataset_results in results.items():
            print(f"\nDATASET: {dataset_name}")
            print("-" * 80)

            for model_name, model_experiments in dataset_results.items():
                print(f"\nModel: {model_name}")

                # Create header
                header = ["Type", "Users%", "Community", "PowerNodes"] + [f"{metric}@{topk}" for metric in metrics]
                print(f"{'Type':<12} {'Users%':<8} {'Community':<12} {'PowerNodes':<12}", end="")
                for metric in metrics:
                    print(f"{metric}@{topk:<12}", end="")
                print()
                print("-" * (12 + 8 + 12 + 12 + 12 * len(metrics)))

                # Print each experiment
                for exp in model_experiments:
                    exp_type = exp['experiment_type']
                    users_perc = str(exp['users_dec_perc_suppr']) if exp['users_dec_perc_suppr'] != 'N/A' else 'N/A'
                    community = str(exp['community_suppression']) if exp['community_suppression'] != 'N/A' else 'N/A'
                    power_nodes = str(exp['suppress_power_nodes_first']) if exp[
                                                                                'suppress_power_nodes_first'] != 'N/A' else 'N/A'

                    print(f"{exp_type:<12} {users_perc:<8} {community:<12} {power_nodes:<12}", end="")

                    for metric_col in metric_columns:
                        value = exp[metric_col]
                        if isinstance(value, float):
                            print(f"{value:<12.4f}", end="")
                        else:
                            print(f"{str(value):<12}", end="")
                    print()
                print()

    # Save results
    if output_format in ["json", "both"]:
        # Save as JSON
        output_path = Path(csv_path).parent / f"formatted_results_top{topk}.json"
        with open(output_path, 'w') as f:
            json.dump(formatted_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    if output_format in ["table", "both"]:
        # Save as readable text table
        output_path = Path(csv_path).parent / f"formatted_results_top{topk}.txt"
        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(f"EXPERIMENT RESULTS - Top@{topk} Metrics\n")
            f.write("=" * 100 + "\n")

            for dataset_name, dataset_results in results.items():
                f.write(f"\nDATASET: {dataset_name}\n")
                f.write("-" * 80 + "\n")

                for model_name, model_experiments in dataset_results.items():
                    f.write(f"\nModel: {model_name}\n")

                    # Write header
                    f.write(f"{'Type':<12} {'Users%':<8} {'Community':<12} {'PowerNodes':<12}")
                    for metric in metrics:
                        f.write(f"{metric}@{topk:<12}")
                    f.write("\n")
                    f.write("-" * (12 + 8 + 12 + 12 + 12 * len(metrics)) + "\n")

                    # Write each experiment
                    for exp in model_experiments:
                        exp_type = exp['experiment_type']
                        users_perc = str(exp['users_dec_perc_suppr']) if exp['users_dec_perc_suppr'] != 'N/A' else 'N/A'
                        community = str(exp['community_suppression']) if exp[
                                                                             'community_suppression'] != 'N/A' else 'N/A'
                        power_nodes = str(exp['suppress_power_nodes_first']) if exp[
                                                                                    'suppress_power_nodes_first'] != 'N/A' else 'N/A'

                        f.write(f"{exp_type:<12} {users_perc:<8} {community:<12} {power_nodes:<12}")

                        for metric_col in metric_columns:
                            value = exp[metric_col]
                            if isinstance(value, float):
                                f.write(f"{value:<12.4f}")
                            else:
                                f.write(f"{str(value):<12}")
                        f.write("\n")
                    f.write("\n")

        print(f"Table saved to: {output_path}")

    return formatted_results


# Example 1: Extract NDCG and Recall at top@10
results = extract_experiment_results(
    # metrics=["ndcg", "user_community_bias"],
    topk=20,
    output_format="both",
    csv_path="logs/full_experiments_best_biases_over_hyperparams/full_experiments_best_biases_over_hyperparams.csv"
)

# Example 2: Extract different metrics at top@20
# results = extract_experiment_results(
#     metrics=["ndcg", "recall", "item_coverage"],
#     topk=20,
#     output_format="table"
# )

# Example 3: Extract diversity metrics at top@50
# results = extract_experiment_results(
#     metrics=["intra_list_diversity", "simpson_index_genre", "user_community_bias"],
#     topk=50,
#     output_format="json"
# )