import json
import os
import glob
import pandas as pd
from typing import List, Dict, Optional, Union


def extract_metrics_from_logs(
        log_directory: str = "logs/no suppression/",
        metric_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Extract performance metrics from log files.

    Parameters:
    -----------
    log_directory : str
        Directory containing log files
    metric_names : List[str] or None
        List of metric names to extract. If None, extract all metrics.
    k_values : List[int] or None
        List of k values (10, 20, 50, 100) to extract. If None, extract all.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing extracted metrics
    """

    # Default k values if not specified
    all_k_values = [10, 20, 50, 100]
    if k_values is None:
        k_values = all_k_values
    else:
        # Validate k values
        k_values = [k for k in k_values if k in all_k_values]

    # Find all log files in the directory
    log_files = glob.glob(os.path.join(log_directory, "*.log"))

    results = []

    for log_file in log_files:
        try:
            # Extract model and dataset info from filename
            filename = os.path.basename(log_file)

            # Read the file and find the last line with cv_avg
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Find the last line containing cv_avg (cross-validation average)
            cv_avg_line = None
            for line in reversed(lines):
                if '"stage":"cv_avg"' in line:
                    cv_avg_line = line.strip()
                    break

            if cv_avg_line is None:
                print(f"Warning: No cv_avg found in {filename}")
                continue

            # Parse the JSON line
            data = json.loads(cv_avg_line)

            # Extract experiment info
            experiment_id = data.get('experiment_id', '')

            # Parse experiment_id to get dataset and model
            parts = experiment_id.split('_')
            if len(parts) >= 2:
                dataset = parts[0]
                model = parts[1]
            else:
                dataset = 'unknown'
                model = experiment_id

            # Extract metrics for each k value
            metrics_data = data.get('metrics', {})

            for k in k_values:
                k_key = f"top@{k}"
                if k_key in metrics_data:
                    k_metrics = metrics_data[k_key]

                    # Filter metrics if specified
                    if metric_names is not None:
                        k_metrics = {m: v for m, v in k_metrics.items() if m in metric_names}

                    # Create a row for each metric
                    for metric_name, value in k_metrics.items():
                        results.append({
                            'dataset': dataset,
                            'model': model,
                            'k': k,
                            'metric': metric_name,
                            'value': value,
                            'experiment_id': experiment_id
                        })

        except Exception as e:
            print(f"Error processing {log_file}: {str(e)}")

    # Create DataFrame
    df = pd.DataFrame(results)

    return df


def format_latex_table(
        df: pd.DataFrame,
        metric_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        precision: int = 4
) -> str:
    """
    Format the metrics DataFrame as a LaTeX table.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing metrics
    metric_names : List[str] or None
        Metrics to include in the table
    k_values : List[int] or None
        k values to include in the table
    precision : int
        Number of decimal places for values

    Returns:
    --------
    str
        LaTeX table code
    """

    if df.empty:
        return "% No data found"

    # Filter data if needed
    if metric_names is not None:
        df = df[df['metric'].isin(metric_names)]
    if k_values is not None:
        df = df[df['k'].isin(k_values)]

    # Determine table structure based on what's being displayed
    unique_metrics = df['metric'].unique()
    unique_k = sorted(df['k'].unique())
    unique_models = df['model'].unique()
    unique_datasets = df['dataset'].unique()

    # Case 1: Single metric, multiple k values
    if len(unique_metrics) == 1 and len(unique_k) > 1:
        metric = unique_metrics[0]
        pivot_df = df[df['metric'] == metric].pivot_table(
            index=['dataset', 'model'],
            columns='k',
            values='value'
        )

        # Create LaTeX table
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{metric} performance across different k values}}\n"
        latex += "\\begin{tabular}{ll" + "r" * len(unique_k) + "}\n"
        latex += "\\toprule\n"
        latex += "Dataset & Model & " + " & ".join([f"k={k}" for k in unique_k]) + " \\\\\n"
        latex += "\\midrule\n"

        for (dataset, model), row in pivot_df.iterrows():
            values = [f"{row[k]:.{precision}f}" if pd.notna(row[k]) else "-" for k in unique_k]
            latex += f"{dataset} & {model} & " + " & ".join(values) + " \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"

    # Case 2: Single k value, multiple metrics
    elif len(unique_k) == 1 and len(unique_metrics) > 1:
        k = unique_k[0]
        pivot_df = df[df['k'] == k].pivot_table(
            index=['dataset', 'model'],
            columns='metric',
            values='value'
        )

        # Create LaTeX table
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{Performance metrics at k={k}}}\n"
        latex += "\\begin{tabular}{ll" + "r" * len(unique_metrics) + "}\n"
        latex += "\\toprule\n"
        latex += "Dataset & Model & " + " & ".join(unique_metrics) + " \\\\\n"
        latex += "\\midrule\n"

        for (dataset, model), row in pivot_df.iterrows():
            values = [f"{row[m]:.{precision}f}" if pd.notna(row[m]) else "-" for m in unique_metrics]
            latex += f"{dataset} & {model} & " + " & ".join(values) + " \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"

    # Case 3: Single metric and single k
    elif len(unique_metrics) == 1 and len(unique_k) == 1:
        metric = unique_metrics[0]
        k = unique_k[0]

        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{metric} at k={k}}}\n"
        latex += "\\begin{tabular}{llr}\n"
        latex += "\\toprule\n"
        latex += f"Dataset & Model & {metric} \\\\\n"
        latex += "\\midrule\n"

        for _, row in df.iterrows():
            latex += f"{row['dataset']} & {row['model']} & {row['value']:.{precision}f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"

    # Case 4: Multiple metrics and k values - create separate tables
    else:
        latex = ""
        for metric in unique_metrics:
            metric_df = df[df['metric'] == metric]
            pivot_df = metric_df.pivot_table(
                index=['dataset', 'model'],
                columns='k',
                values='value'
            )

            latex += "\\begin{table}[htbp]\n"
            latex += "\\centering\n"
            latex += f"\\caption{{{metric} performance across different k values}}\n"
            latex += "\\begin{tabular}{ll" + "r" * len(unique_k) + "}\n"
            latex += "\\toprule\n"
            latex += "Dataset & Model & " + " & ".join([f"k={k}" for k in unique_k]) + " \\\\\n"
            latex += "\\midrule\n"

            for (dataset, model), row in pivot_df.iterrows():
                values = [f"{row[k]:.{precision}f}" if k in row and pd.notna(row[k]) else "-" for k in unique_k]
                latex += f"{dataset} & {model} & " + " & ".join(values) + " \\\\\n"

            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"
            latex += "\\end{table}\n\n"

    return latex


def scrape_all_logs(
        log_directory: str = "logs/no suppression",
        metric_names: Optional[Union[str, List[str]]] = None,
        k_values: Optional[Union[int, List[int]]] = None,
        output_file: Optional[str] = None
):
    """
    Main function to extract metrics and create LaTeX tables.

    Parameters:
    -----------
    log_directory : str
        Directory containing log files
    metric_names : str, List[str], or None
        Metric names to extract. Can be a single metric, list of metrics, or None for all.
    k_values : int, List[int], or None
        k values to extract. Can be a single k, list of k values, or None for all.
    output_file : str or None
        If provided, save LaTeX table to this file
    """

    # Handle single metric/k inputs
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    if isinstance(k_values, int):
        k_values = [k_values]

    # Extract metrics from logs
    print(f"Extracting metrics from {log_directory}...")
    df = extract_metrics_from_logs(log_directory, metric_names, k_values)

    if df.empty:
        print("No metrics found!")
        return

    print(f"Found {len(df)} metric entries")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Models: {df['model'].unique()}")
    print(f"Metrics: {df['metric'].unique()}")
    print(f"k values: {sorted(df['k'].unique())}")

    # Generate LaTeX table
    latex_table = format_latex_table(df, metric_names, k_values)

    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to {output_file}")
    else:
        print("\nLaTeX Table:")
        print(latex_table)

    return df, latex_table


# Example 1: Extract all metrics for all k values
df, latex = scrape_all_logs(metric_names='user_community_bias', log_directory="logs/no suppression/", output_file='logs/no suppression/no_suppression.tex', k_values=[10, 20, 50, 100])
print("\nDataFrame:")
print(df)
print("\nLaTeX Table:")
print(latex)
# Example 2: Extract only NDCG for all k values
# df, latex = main(metric_names="ndcg")

# Example 3: Extract all metrics for k=10 only
# df, latex = main(k_values=10)

# Example 4: Extract NDCG and Recall for k=10 and k=20
# df, latex = main(metric_names=["ndcg", "recall"], k_values=[10, 20])

# Example 5: Save to file
# df, latex = main(metric_names="ndcg", output_file="ndcg_results.tex")