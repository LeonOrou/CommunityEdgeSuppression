import os
import wandb
import pandas as pd
from datetime import datetime
import re
import json
import glob
from tabulate import tabulate
import yaml


def get_hyperparameter_results():
    print("Extracting runs from local wandb/ folder...")
    runs = []
    run_folders = glob.glob(os.path.join("wandb", "run-*"))
    print(f"Found {len(run_folders)} total local runs")
    for folder in run_folders:
        config_path = os.path.join(folder, "files/config.yaml")
        summary_path = os.path.join(folder, "files/wandb-summary.json")
        if not os.path.exists(config_path) or not os.path.exists(summary_path):
            continue
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        with open(summary_path, "r") as f:
            summary = json.load(f)
        # Use file modification time as created_at timestamp
        created_at = datetime.fromtimestamp(os.path.getmtime(config_path))
        run = {"config": config, "summary": summary, "created_at": created_at}
        runs.append(run)

    # Store runs by model
    model_runs = {
        'LightGCN': [],
        'ItemKNN': [],
        'MultiVAE': []
    }

    # Process each run
    for run in runs:
        config = run["config"]
        model_name = config.get('model')['value']
        if model_name not in model_runs:
            continue

        # Extract the metrics
        summary = dict(run["summary"])
        test_ndcg = None
        item_coverage = None
        gini_index = None
        avg_popularity = None

        for key in summary:
            key_lower = key.lower()
            if 'test_ndcg@10' in key_lower:
                test_ndcg = summary[key]
            elif 'test_itemcoverage@10' in key_lower:
                item_coverage = summary[key]
            elif 'test_giniindex@10' in key_lower:
                gini_index = summary[key]
            elif 'test_averagepopularity@10' in key_lower:
                avg_popularity = summary[key]

        if test_ndcg is None:
            continue

        # Extract parameters based on model type
        params = {
            'test_ndcg': test_ndcg,
            'test_item_coverage': item_coverage,
            'test_gini_index': gini_index,
            'test_avg_popularity': avg_popularity
        }

        # Common parameters
        common_keys = ['learning_rate', 'embedding_size', 'scheduler']
        for key in common_keys:
            if key in config:
                params[key] = config.get(key)['value']

        # Model-specific parameters
        if model_name == 'LightGCN':
            if 'n_layers' in config:
                params['n_layers'] = config.get('n_layers')['value']
        elif model_name == 'ItemKNN':
            if 'k_values' in config:
                params['k_values'] = config.get('k_values')['value']
            if 'shrink' in config:
                params['shrink'] = config.get('shrink')['value']
        elif model_name == 'MultiVAE':
            for key in ['hidden_dimension', 'latent_dimension', 'dropout_prob', 'anneal_cap']:
                if key in config:
                    params[key] = config.get(key)['value']

        # Add timestamp and run info
        params['created_at'] = run["created_at"]
        model_runs[model_name].append(params)

    # Process results for each model
    results = {}
    for model_name, runs in model_runs.items():
        if not runs:
            continue

        # Convert to DataFrame
        df = pd.DataFrame(runs)

        # Define parameter columns to identify unique configurations
        if model_name == 'LightGCN':
            param_cols = ['learning_rate', 'embedding_size', 'n_layers']
        elif model_name == 'ItemKNN':
            param_cols = ['k_values', 'shrink']
        else:  # MultiVAE
            param_cols = ['hidden_dimension', 'latent_dimension', 'dropout_prob', 'anneal_cap']

        # Keep only columns that exist in the dataframe
        param_cols = [col for col in param_cols if col in df.columns]

        # Keep newest run for each unique parameter combination
        df = df.sort_values('created_at', ascending=False)
        df = df.drop_duplicates(subset=param_cols, keep='first')

        # Sort by test NDCG in descending order
        df = df.sort_values('test_ndcg', ascending=False)

        results[model_name] = df

    return results


def display_results(results):
    for model_name, df in results.items():
        if df.empty:
            print(f"No results for {model_name}")
            continue

        print(f"\n{model_name} Results (top 10):")

        # Select columns to display based on the model type
        if model_name == 'LightGCN':
            display_cols = ['n_layers', 'embedding_size', 'learning_rate', 'scheduler',
                            'test_ndcg', 'test_item_coverage', 'test_gini_index', 'test_avg_popularity']
        elif model_name == 'ItemKNN':
            display_cols = ['k_values', 'shrink', 'test_ndcg', 'test_item_coverage', 'test_gini_index', 'test_avg_popularity']
        else:  # MultiVAE
            display_cols = ['hidden_dimension', 'latent_dimension', 'dropout_prob',
                            'anneal_cap', 'test_ndcg', 'test_item_coverage', 'test_gini_index', 'test_avg_popularity']

        # Only include columns that exist in the dataframe
        display_cols = [col for col in display_cols if col in df.columns]

        # Display top 10 results
        print(tabulate(df[display_cols].head(10), headers='keys', tablefmt='github', floatfmt=".4f"))

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%n_categories%S")
        # Create directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        filename = f"results/{model_name}_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


results = get_hyperparameter_results()
display_results(results)

