import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


def parse_log_file(filepath):
    """Parse a single log file and extract configuration and metrics."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse configuration from first line
    config_line = lines[0].strip()
    config = json.loads(config_line)

    # Find cv_avg metrics (last metrics_report)
    cv_avg_metrics = None
    for line in reversed(lines):
        if '"stage":"cv_avg"' in line and '"event":"metrics_report"' in line:
            cv_avg_metrics = json.loads(line.strip())
            break

    if not cv_avg_metrics:
        return None

    return {
        'model_name': config['model_name'],
        'dataset_name': config['dataset_name'],
        'use_suppression': config.get('use_suppression', False),
        'community_suppression': config.get('community_suppression', 0),
        'users_top_percent': config.get('users_top_percent', 0),
        'users_dec_perc_suppr': config.get('users_dec_perc_suppr', 0),
        'suppress_power_nodes_first': config.get('suppress_power_nodes_first', False),
        'metrics': cv_avg_metrics['metrics']['top@10']
    }


def get_config_key(result):
    """Create a unique key for each configuration."""
    return (
        result['use_suppression'],
        result['community_suppression'],
        result['users_top_percent'],
        result['users_dec_perc_suppr'],
        result['suppress_power_nodes_first']
    )


def format_config_label(config_key):
    """Format configuration key into readable label."""
    use_suppr, comm_suppr, users_top, users_dec, power_first = config_key

    if not use_suppr:
        return "Baseline"

    parts = []
    if comm_suppr > 0:
        parts.append(f"CS={comm_suppr}")
    if users_top > 0:
        parts.append(f"UTP={users_top}")
    if users_dec > 0:
        parts.append(f"UDS={users_dec}")
    if power_first:
        parts.append("PF")

    return " ".join(parts) if parts else "Suppression"


def calculate_percentage_change(value, baseline):
    """Calculate percentage change from baseline."""
    if baseline == 0:
        return 0
    return ((value - baseline) / baseline) * 100


def get_color_code(pct_change, lower_is_better=False):
    """Get LaTeX color code based on percentage change."""
    if lower_is_better:
        pct_change = -pct_change  # Invert for metrics where lower is better

    if pct_change >= 5:
        return r'\cellcolor{green!40}'  # Green
    elif pct_change > 0:
        return r'\cellcolor{green!20}'  # Light green
    elif pct_change <= -5:
        return r'\cellcolor{red!40}'  # Red
    else:
        return r'\cellcolor{red!20}'  # Light red


def format_metric_value(value, baseline_value, is_baseline, is_best, lower_is_better=False):
    """Format metric value with color and bold if needed."""
    formatted = f"{value:.4f}"

    if not is_baseline:
        pct_change = calculate_percentage_change(value, baseline_value)
        color = get_color_code(pct_change, lower_is_better)
        formatted = f"{color} {formatted}"

    if is_best:
        formatted = f"\\textbf{{{formatted}}}"

    return formatted


def format_metric_name(metric):
    """Format metric name for display."""
    replacements = {
        'ndcg': 'NDCG',
        'recall': 'Recall',
        'precision': 'Precision',
        'mrr': 'MRR',
        'hit_rate': 'Hit Rate',
        'item_coverage': 'Item Coverage',
        'gini_index': 'Gini Index',
        'average_rec_popularity': 'Avg. Rec. Pop.',
        'popularity_lift': 'Pop. Lift',
        'pop_miscalibration': 'Pop. Miscal.',
        'simpson_index_genre': 'Simpson Index',
        'intra_list_diversity': 'Intra-list Div.',
        'normalized_genre_entropy': 'Genre Entropy',
        'unique_genres_count': 'Unique Genres',
        'user_community_bias': 'User Comm. Bias'
    }
    return replacements.get(metric, metric.replace('_', ' ').title())


def full_table():
    """Generate full LaTeX table comparing all test settings."""
    # Path to log files
    log_dir = Path("logs/full experiments/")

    # Parse all log files
    results = []
    for log_file in log_dir.glob("*.log"):
        result = parse_log_file(log_file)
        if result:
            results.append(result)

    if not results:
        print("No log files found or parsed successfully!")
        return

    # Organize data by model, dataset, and configuration
    data = {}
    for result in results:
        model = result['model_name']
        dataset = result['dataset_name']
        config_key = get_config_key(result)

        if model not in data:
            data[model] = {}
        if dataset not in data[model]:
            data[model][dataset] = {}

        data[model][dataset][config_key] = result['metrics']

    # Metrics to extract
    metrics = ['ndcg', 'recall', 'precision', 'mrr', 'hit_rate', 'item_coverage',
               'gini_index', 'average_rec_popularity', 'popularity_lift',
               'pop_miscalibration', 'simpson_index_genre', 'intra_list_diversity',
               'normalized_genre_entropy', 'unique_genres_count', 'user_community_bias']

    # Metrics where lower values are better
    lower_is_better_metrics = ['gini_index', 'average_rec_popularity', 'popularity_lift',
                               'pop_miscalibration', 'user_community_bias']

    # Models and datasets in order
    models = ['ItemKNN', 'MultVAE', 'LightGCN']
    datasets = ['ML-100K', 'ML-1M', 'Last.FM']

    # Map dataset names to their keys in data
    dataset_keys = {
        'ML-100K': ['ml-100k', 'ML-100K'],
        'ML-1M': ['ml-1m', 'ML-1M'],
        'Last.FM': ['lastfm', 'Last.FM', 'last.fm']
    }

    # Build LaTeX table
    latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{adjustbox}
\usepackage{array}

\begin{document}

\begin{table}[htbp]
\centering
\caption{Experimental Results at top@10 - All Configurations}
\begin{adjustbox}{width=\textwidth,center}
\begin{tabular}{ll"""

    # Add column specifications for metrics
    latex += 'c' * len(metrics) + r"""}
\toprule
"""

    # Process each dataset
    for dataset_idx, dataset in enumerate(datasets):
        # Add dataset header
        latex += f"\\multicolumn{{2}}{{l}}{{\\textbf{{{dataset}}}}} & "
        latex += " & ".join([f"\\rotatebox{{90}}{{{format_metric_name(m)}}}" for m in metrics])
        latex += r" \\" + "\n\\midrule\n"

        # Find dataset key
        dataset_key = None
        for model in models:
            if model in data:
                for d_key in dataset_keys.get(dataset, [dataset.lower()]):
                    if d_key in data[model]:
                        dataset_key = d_key
                        break
                if dataset_key:
                    break

        if not dataset_key:
            continue

        # Process each model
        for model_idx, model in enumerate(models):
            if model not in data or dataset_key not in data[model]:
                continue

            model_data = data[model][dataset_key]

            # Get all configurations for this model/dataset
            config_keys = sorted(model_data.keys())

            # Find baseline configuration
            baseline_key = (False, 0, 0, 0, False)
            if baseline_key not in config_keys:
                # Try to find any config with use_suppression=False
                baseline_key = next((k for k in config_keys if not k[0]), None)

            # Find best values for each metric in this model/dataset
            best_values = {}
            for metric in metrics:
                values = []
                for config_key in config_keys:
                    if metric in model_data[config_key]:
                        values.append(model_data[config_key][metric])

                if values:
                    if metric in lower_is_better_metrics:
                        best_values[metric] = min(values)
                    else:
                        best_values[metric] = max(values)

            # Process each configuration
            for config_idx, config_key in enumerate(config_keys):
                if config_idx == 0:
                    row = [f"\\multirow{{{len(config_keys)}}}{{*}}{{{model}}}",
                           format_config_label(config_key)]
                else:
                    row = ["", format_config_label(config_key)]

                config_metrics = model_data[config_key]
                baseline_metrics = model_data.get(baseline_key, config_metrics)

                for metric in metrics:
                    if metric in config_metrics:
                        is_best = config_metrics[metric] == best_values.get(metric, None)
                        is_baseline = (config_key == baseline_key)
                        lower_is_better = metric in lower_is_better_metrics
                        value = format_metric_value(config_metrics[metric],
                                                    baseline_metrics.get(metric, config_metrics[metric]),
                                                    is_baseline, is_best, lower_is_better)
                        row.append(value)
                    else:
                        row.append("-")

                latex += " & ".join(row) + r" \\" + "\n"

            # Add single line after each model except the last
            if model_idx < len(models) - 1:
                latex += r"\cmidrule{1-" + str(2 + len(metrics)) + "}\n"

        # Add double line after each dataset except the last
        if dataset_idx < len(datasets) - 1:
            latex += r"\midrule\midrule" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{adjustbox}

\vspace{0.5cm}
\footnotesize
\textbf{Configuration abbreviations:} 
CS = Community Suppression, 
UTP = Users Top Percent, 
UDS = Users Decrease Percent Suppression, 
PF = Suppress Power Nodes First

\end{table}

\end{document}
"""

    # Save LaTeX file
    output_path = log_dir / "results_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to: {output_path}")
    print("\nLaTeX code preview (first 100 lines):")
    print('\n'.join(latex.split('\n')[:100]))

    # Also create a simplified version without document preamble
    table_start = latex.find(r'\begin{table}')
    table_end = latex.find(r'\end{document}')
    table_only = latex[table_start:table_end]

    with open(log_dir / "results_table_only.tex", 'w') as f:
        f.write(table_only)

    print(f"\nTable-only version saved to: {log_dir / 'results_table_only.tex'}")

    # Print summary of parsed data
    print("\n=== Summary of parsed data ===")
    for model in data:
        print(f"\nModel: {model}")
        for dataset in data[model]:
            print(f"  Dataset: {dataset}")
            for config in data[model][dataset]:
                print(f"    Config {format_config_label(config)}: {len(data[model][dataset][config])} metrics")