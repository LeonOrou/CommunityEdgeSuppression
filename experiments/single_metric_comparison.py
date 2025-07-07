import json
import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path


def extract_log_data(log_file_path):
    """Extract experiment configuration and results from a log file."""
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        # Parse first line (configuration)
        config_line = lines[0].strip()
        config = json.loads(config_line)

        # Parse last line (cv_avg results)
        cv_avg_line = lines[-1].strip()
        cv_avg_data = json.loads(cv_avg_line)

        # Extract key configuration parameters
        extracted_data = {
            'model_name': config.get('model_name', ''),
            'dataset_name': config.get('dataset_name', ''),
            'users_top_percent': config.get('users_top_percent', 0),
            'users_dec_perc_suppr': config.get('users_dec_perc_suppr', 0),
            'community_suppression': config.get('community_suppression', 0),
            'suppress_power_nodes_first': config.get('suppress_power_nodes_first', False),
            'use_suppression': config.get('use_suppression', False),
            'experiment_id': config.get('experiment_id', ''),
            'log_file': os.path.basename(log_file_path)
        }

        # Extract metrics for each top@X
        if 'metrics' in cv_avg_data and cv_avg_data['stage'] == 'cv_avg':
            metrics = cv_avg_data['metrics']
            for top_k in metrics:
                for metric_name, metric_value in metrics[top_k].items():
                    extracted_data[f'{top_k}_{metric_name}'] = metric_value

        return extracted_data

    except Exception as e:
        print(f"Error processing {log_file_path}: {e}")
        return None


def create_single_metric_table(df, top_k, metric_name, output_path):
    """Create a focused table for a single metric at a specific top@K value."""

    # Define models and datasets
    models = ['ItemKNN', 'MultiVAE', 'LightGCN']
    datasets = ['ml-100k', 'ml-1m', 'lastfm']

    # The specific metric column we're analyzing
    metric_col = f'top@{top_k}_{metric_name}'

    if metric_col not in df.columns:
        raise ValueError(f"Metric {metric_col} not found in data. Available columns: {df.columns.tolist()}")

    # Create baseline dictionary
    baselines = {}
    baseline_df = df[df['use_suppression'] == False]

    for dataset in datasets:
        dataset_baselines = baseline_df[baseline_df['dataset_name'] == dataset]
        if not dataset_baselines.empty:
            baseline_value = dataset_baselines.iloc[0][metric_col]
            baselines[dataset] = baseline_value

    # Function to determine color based on improvement percentage
    def get_color(value, baseline_value):
        if pd.isna(value) or pd.isna(baseline_value) or baseline_value == 0:
            return ''

        improvement = (value - baseline_value) / baseline_value * 100

        if improvement < 5:
            return 'background-color: #90EE90'  # Light green
        elif improvement < 0:
            return 'background-color: #008000; color: white'  # Green
        elif improvement > -5:
            return 'background-color: #FF0000; color: white'  # Red
        elif improvement > 0:
            return 'background-color: #FFB6C1'  # Light red
        else:
            return ''

    # Create HTML table
    html_content = f"""
    <html>
    <head>
        <style>
            table {{ border-collapse: collapse; width: 100%; font-size: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .baseline {{ font-weight: bold; background-color: #e6f3ff; }}
            .best {{ font-weight: bold; text-decoration: underline; }}
            .model-header {{ background-color: #d9d9d9; font-weight: bold; }}
            .dataset-header {{ background-color: #f0f0f0; font-weight: bold; }}
        </style>
    </head>
    <body>
    <h2>Recommendation System Results: {metric_name.upper()} @ Top-{top_k}</h2>
    <table>
    """

    # Create header - datasets as columns, hyperparameter combinations as rows
    html_content += "<tr><th>Model</th><th>Hyperparameters</th>"
    for dataset in datasets:
        html_content += f"<th class='dataset-header'>{dataset}</th>"
    html_content += "</tr>\n"

    # Find best values for each dataset
    best_values = {}
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset]
        if not dataset_df.empty and metric_col in dataset_df.columns:
            best_values[dataset] = dataset_df[metric_col].max()

    # Group by model and create rows
    for model in models:
        model_df = df[df['model_name'] == model]
        if model_df.empty:
            continue

        # Get unique hyperparameter combinations for this model
        hyperparam_cols = ['use_suppression', 'users_top_percent', 'users_dec_perc_suppr',
                           'community_suppression', 'suppress_power_nodes_first']
        unique_hyperparams = model_df[hyperparam_cols].drop_duplicates()

        model_row_count = len(unique_hyperparams)
        first_row = True

        for idx, (_, hyperparam_row) in enumerate(unique_hyperparams.iterrows()):
            html_content += "<tr>"

            # Model name (span across multiple rows)
            if first_row:
                html_content += f'<td class="model-header" rowspan="{model_row_count}">{model}</td>'
                first_row = False

            # Hyperparameters column
            use_supp = hyperparam_row['use_suppression']
            if use_supp:
                hyperparam_str = f"Supp=True, UT={hyperparam_row['users_top_percent']}, " \
                                 f"UD={hyperparam_row['users_dec_perc_suppr']}, " \
                                 f"CS={hyperparam_row['community_suppression']}, " \
                                 f"SPF={hyperparam_row['suppress_power_nodes_first']}"
            else:
                hyperparam_str = "Baseline (Supp=False)"

            if use_supp:
                html_content += f'<td>{hyperparam_str}</td>'
            else:
                html_content += f'<td class="baseline">{hyperparam_str}</td>'

            # Values for each dataset
            for dataset in datasets:
                # Find the row with this model, dataset, and hyperparameter combination
                condition = (model_df['dataset_name'] == dataset)
                for col in hyperparam_cols:
                    condition = condition & (model_df[col] == hyperparam_row[col])

                matching_rows = model_df[condition]

                if not matching_rows.empty:
                    value = matching_rows.iloc[0][metric_col]
                    cell_class = ""
                    style = ""

                    # Color coding based on baseline comparison
                    if dataset in baselines:
                        baseline_value = baselines[dataset]
                        if not use_supp:
                            cell_class = "baseline"
                        else:
                            style = get_color(value, baseline_value)

                    # Mark best value
                    if dataset in best_values and abs(value - best_values[dataset]) < 1e-10:
                        cell_class += " best"

                    display_value = f"{value:.4f}" if not pd.isna(value) else "N/A"
                    html_content += f'<td class="{cell_class}" style="{style}">{display_value}</td>'
                else:
                    html_content += '<td>N/A</td>'

            html_content += "</tr>\n"

    html_content += """
    </table>
    <br>
    <p><strong>Color Legend:</strong></p>
    <p><span style="background-color: #90EE90; padding: 2px;">Light Green</span>: Improvement > 5%</p>
    <p><span style="background-color: #008000; color: white; padding: 2px;">Green</span>: Improvement 0-5%</p>
    <p><span style="background-color: #FFB6C1; padding: 2px;">Light Red</span>: Decline 0-5%</p>
    <p><span style="background-color: #FF0000; color: white; padding: 2px;">Red</span>: Decline > 5%</p>
    <p><span style="background-color: #e6f3ff; padding: 2px;">Blue Background</span>: Baseline (use_suppression=False)</p>
    <p><strong>Bold & Underlined</strong>: Best result for dataset</p>
    </body>
    </html>
    """

    # Save HTML file
    html_file = output_path.replace('.csv', f'_{metric_name}_top{top_k}.html')
    with open(html_file, 'w') as f:
        f.write(html_content)

    print(f"Single metric table saved to: {html_file}")
    return html_file


def single_metric_table(top_k=10, metric_name='ndcg', path='logs/full_experiments_best_biases/'):
    """
    Create a focused table for a single metric at a specific top@K value.

    Parameters:
    top_k (int): The top-K value (e.g., 10 for top@10)
    metric_name (str): The metric to analyze (e.g., 'ndcg', 'recall', 'precision', etc.)
    """
    # Define the logs folder path
    logs_folder = path

    # Check if folder exists
    if not os.path.exists(logs_folder):
        print(f"Folder {logs_folder} does not exist. Creating it...")
        os.makedirs(logs_folder, exist_ok=True)
        print("Please place your .log files in the created folder and run the script again.")
        return

    # Find all .log files
    log_files = glob.glob(os.path.join(logs_folder, "*.log"))

    if not log_files:
        print(f"No .log files found in {logs_folder}")
        return

    print(f"Found {len(log_files)} log files")
    print(f"Creating table for metric: {metric_name} @ top-{top_k}")

    # Extract data from all log files
    all_data = []
    for log_file in log_files:
        print(f"Processing: {log_file}")
        data = extract_log_data(log_file)
        if data:
            all_data.append(data)

    if not all_data:
        print("No valid data extracted from log files")
        return

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Sort by model, dataset, and suppression settings
    df = df.sort_values(['model_name', 'dataset_name', 'use_suppression', 'users_top_percent', 'community_suppression'])

    # Save raw DataFrame
    output_path = os.path.join(logs_folder, f"experiment_results_{metric_name}_top{top_k}.csv")
    df.to_csv(output_path, index=False)
    print(f"Raw DataFrame saved to: {output_path}")

    # Print basic info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Models found: {df['model_name'].unique()}")
    print(f"Datasets found: {df['dataset_name'].unique()}")
    print(f"Use suppression values: {df['use_suppression'].unique()}")

    # Check if the requested metric exists
    metric_col = f'top@{top_k}_{metric_name}'
    if metric_col not in df.columns:
        print(f"\nError: Metric '{metric_col}' not found!")
        print("Available metrics:")
        metric_cols = [col for col in df.columns if 'top@' in col]
        for col in sorted(metric_cols):
            print(f"  {col}")
        return

    # Display first few rows
    print(f"\nFirst 5 rows of {metric_col}:")
    print(df[['model_name', 'dataset_name', 'use_suppression', metric_col]].head())

    # Create focused single metric table
    html_file = create_single_metric_table(df, top_k, metric_name, output_path)

    # Print summary statistics for the specific metric
    print(f"\nSummary statistics for {metric_col}:")
    print(df.groupby(['model_name', 'dataset_name', 'use_suppression'])[metric_col].agg(['count', 'mean', 'std']).round(
        4))

    print(f"\nProcessing complete! Check {html_file} for the styled results table.")

    return df, html_file


single_metric_table(top_k=20,
                    metric_name='ndcg',
                    path='logs/full_experiments_best_biases_over_hyperparams/')  # Example usage, can be adjusted as needed

