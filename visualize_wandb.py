import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Add this after your existing visualizations

# Create plots specifically for community_dropout_strength impact
def plot_community_dropout_strength_impact(df, output_dir):
    # Get all metrics (removed 'params' from the exclusion list)
    metric_cols = [col for col in df.columns]

    # 1. Bar plots grouped by community_dropout_strength
    dropout_values = sorted(df['community_dropout_strength'].unique())
    for metric in metric_cols:
        plt.figure(figsize=(12, 8))

        # Group by strength and calculate mean
        strength_impact = df.groupby('community_dropout_strength').mean().reset_index()

        # Create bar plot
        ax = sns.barplot(x='community_dropout_strength', y=metric, data=strength_impact,
                         palette='viridis', order=dropout_values)

        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black',
                        xytext=(0, 5), textcoords='offset points')

        plt.title(f'Impact of Community Dropout Strength on {metric}')
        plt.xlabel('Community Dropout Strength')
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"community_strength_impact_{metric}.png", dpi=300)
        plt.close()

    # 2. Heatmaps showing interaction between community_dropout_strength and users_dec_perc_suppr
    for metric in metric_cols:
        if len(df['community_dropout_strength'].unique()) > 1 and len(df['users_dec_perc_suppr'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            pivot = df.pivot_table(
                index='community_dropout_strength',
                columns='users_dec_perc_suppr',
                values=metric,
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis', cbar_kws={'label': metric})
            plt.title(f'Community Strength vs User Dropout Impact on {metric}')
            plt.xlabel('User Dropout Percentage')
            plt.ylabel('Community Dropout Strength')
            plt.tight_layout()
            plt.savefig(output_dir / f"community_user_interaction_{metric}.png", dpi=300)
            plt.close()

    # 3. Heatmaps showing interaction between community_dropout_strength and items_dec_perc_suppr
    for metric in metric_cols:
        if len(df['community_dropout_strength'].unique()) > 1 and len(df['items_dec_perc_suppr'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            pivot = df.pivot_table(
                index='community_dropout_strength',
                columns='items_dec_perc_suppr',
                values=metric,
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis', cbar_kws={'label': metric})
            plt.title(f'Community Strength vs Item Dropout Impact on {metric}')
            plt.xlabel('Item Dropout Percentage')
            plt.ylabel('Community Dropout Strength')
            plt.tight_layout()
            plt.savefig(output_dir / f"community_item_interaction_{metric}.png", dpi=300)
            plt.close()

    # 4. Box plots to show distribution of metrics by community_dropout_strength
    for metric in metric_cols:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='community_dropout_strength', y=metric, data=df, palette='viridis', order=dropout_values)
        plt.title(f'Distribution of {metric} by Community Dropout Strength')
        plt.xlabel('Community Dropout Strength')
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"community_strength_boxplot_{metric}.png", dpi=300)
        plt.close()


def visualize_wandb_metrics():
    # Initialize WandB API
    api = wandb.Api()

    # Get all runs from your project
    # all_runs = api.runs("leon_orou_projects/RecSys_HPSearch")
    all_runs = api.runs("leon_orou_projects/RecSys_PowerNodeEdgeDropout")

    # Sort all runs by creation time (newest first)
    sorted_runs = sorted(all_runs, key=lambda run: run.created_at, reverse=True)
    
    # Select 27 unique runs (by name)
    unique_names = set()
    runs = []
    
    for run in sorted_runs:
        if run.name not in unique_names:
            runs.append(run)
            unique_names.add(run.name)
            
        if len(runs) >= 27:
            break
            
    print(f"Filtered {len(all_runs)} total runs down to {len(runs)} unique runs (last {len(runs)} entries)")

    # Create output directory
    output_dir = Path("wandb_visualizations")
    output_dir.mkdir(exist_ok=True)

    # Extract run data including summary metrics and history
    run_data = []
    history_data = {}

    for run in runs:
        # Extract configurations
        config = {
            'run_name': run.name,
            'trial': run.config.get('trial'),
            'model': run.config.get('model'),
            'users_dec_perc_suppr': run.config.get('users_dec_perc_suppr'),
            'items_dec_perc_suppr': run.config.get('items_dec_perc_suppr'),
            'community_dropout_strength': run.config.get('community_dropout_strength'),
        }

        # Extract final metrics
        metrics = {k: v for k, v in run.summary.items() if not k.startswith('_')}

        # Combine config and metrics
        run_data.append({**config, **metrics})

        # Get run history for training metrics
        history = run.history()
        history['run_name'] = run.name
        history_data[run.name] = history

    # Create DataFrame for summary metrics
    df = pd.DataFrame(run_data)
    # print(f"Extracted {len(df)} runs with {len(df.columns) - 6} metrics per run")
    #
    # # 1. Bar plots for final metrics by run
    # metric_cols = [col for col in df.columns if col not in ['run_name', 'trial', 'model',
    #                                                         'users_dec_perc_suppr', 'items_dec_perc_suppr',
    #                                                         'community_dropout_strength']]
    #
    # if metric_cols:
    #     # Create combined parameter labels
    #     df['params'] = df.apply(
    #         lambda
    #             x: f"{x['model']}\nu{x['users_dec_perc_suppr']}_i{x['items_dec_perc_suppr']}_c{x['community_dropout_strength']}",
    #         axis=1
    #     )
    #
    #     for i, metric in enumerate(metric_cols):
    #         plt.figure(figsize=(10, 6))
    #         plot_data = df.sort_values(by=metric, ascending=False)
    #         sns.barplot(x='params', y=metric, data=plot_data)
    #         plt.title(f"{metric} by Run Configuration")
    #         plt.xticks(rotation=45, ha='right')
    #         plt.tight_layout()
    #         plt.savefig(output_dir / f"{metric}_by_run.png", dpi=300)
    #         plt.close()
    #
    # # 2. Hyperparameter impact heatmaps
    # for metric in metric_cols:
    #     if len(df['users_dec_perc_suppr'].unique()) > 1 and len(df['items_dec_perc_suppr'].unique()) > 1:
    #         plt.figure(figsize=(8, 6))
    #         pivot = df.pivot_table(
    #             index='users_dec_perc_suppr',
    #             columns='items_dec_perc_suppr',
    #             values=metric,
    #             aggfunc='mean'
    #         )
    #         sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis')
    #         plt.title(f'Impact of Dropout Parameters on {metric}')
    #         plt.tight_layout()
    #         plt.savefig(output_dir / f"{metric}_param_heatmap.png", dpi=300)
    #         plt.close()
    #
    # # 3. Correlation heatmap between metrics
    # if len(metric_cols) > 1:
    #     plt.figure(figsize=(10, 8))
    #     corr = df[metric_cols].corr()
    #     mask = np.triu(np.ones_like(corr, dtype=bool))
    #     sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
    #     plt.title('Correlation Between Metrics')
    #     plt.tight_layout()
    #     plt.savefig(output_dir / "metric_correlation.png", dpi=300)
    #     plt.close()
    #
    # # 4. Training history plots for each run
    # for run_name, history in history_data.items():
    #     # Skip empty histories
    #     if history.empty:
    #         continue
    #
    #     # Get training metrics (exclude non-metrics)
    #     training_metrics = [col for col in history.columns
    #                         if col not in ['_step', '_runtime', '_timestamp', 'run_name']]
    #
    #     # Plot training metrics over time
    #     if training_metrics:
    #         plt.figure(figsize=(12, 8))
    #         for metric in training_metrics:
    #             if metric in history.columns:
    #                 plt.plot(history['_step'], history[metric], label=metric)
    #
    #         plt.title(f'Training Metrics for {run_name}')
    #         plt.xlabel('Step')
    #         plt.ylabel('Value')
    #         plt.legend()
    #         plt.grid(True, linestyle='--', alpha=0.7)
    #         plt.tight_layout()
    #         plt.savefig(output_dir / f"{run_name}_training_history.png", dpi=300)
    #         plt.close()
    #
    # # 5. Combined training metrics across runs
    # # Group common metrics across runs
    # common_metrics = set()
    # for history in history_data.values():
    #     if not history.empty:
    #         common_metrics.update(set(history.columns) - {'_step', '_runtime', '_timestamp', 'run_name'})
    #
    # # Plot each common metric across runs
    # for metric in common_metrics:
    #     plt.figure(figsize=(12, 8))
    #     for run_name, history in history_data.items():
    #         if not history.empty and metric in history:
    #             plt.plot(history['_step'], history[metric], label=run_name)
    #
    #     plt.title(f'{metric} Across All Runs')
    #     plt.xlabel('Step')
    #     plt.ylabel(metric)
    #     plt.legend(loc='best', bbox_to_anchor=(1.0, 1.0))
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig(output_dir / f"comparison_{metric}.png", dpi=300)
    #     plt.close()
    #
    # print(f"Visualizations saved to {output_dir}")

    plot_community_dropout_strength_impact(df, output_dir)

    return df


if __name__ == "__main__":

    df = visualize_wandb_metrics()
    # Also save the data for further analysis
    df.to_csv('wandb_metrics_summary.csv', index=False)

