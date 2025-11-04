"""
Visualization functions for evaluation results.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from core.log_utils import print


def plot_results_boxplot(
    results_dict: Dict[str, Dict[str, any]],
    metric: str = 'auc',
    title: str = None,
    ylabel: str = None,
    figsize: Tuple[int, int] = (14, 8),
    output_path: str = None,
    dpi: int = 300,
    colors: List[str] = None
):
    """
    Create box plot comparing performance across feature sets.

    Args:
        results_dict: Dictionary mapping feature set names to results
        metric: Metric to plot ('auc' or 'accuracy')
        title: Plot title (auto-generated if None)
        ylabel: Y-axis label (auto-generated if None)
        figsize: Figure size
        output_path: Path to save plot (optional)
        dpi: DPI for saved figure
        colors: List of colors for boxes
    """
    # Extract data for plotting
    feature_names = []
    metric_data = []

    for name, results in results_dict.items():
        feature_names.append(name)
        # Handle both standard format ('auc_scores') and model selection format ('all_auc_scores')
        if f'{metric}_scores' in results:
            metric_data.append(results[f'{metric}_scores'])
        elif f'all_{metric}_scores' in results:
            metric_data.append(results[f'all_{metric}_scores'])
        else:
            raise KeyError(f"Neither '{metric}_scores' nor 'all_{metric}_scores' found in results for {name}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot
    bp = ax.boxplot(
        metric_data,
        labels=feature_names,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
    )

    # Color boxes
    if colors is None:
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink',
                  'lavender', 'peachpuff', 'lightcyan'] * 2

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Labels and title
    if ylabel is None:
        ylabel = 'AUC-ROC Score' if metric == 'auc' else 'Accuracy Score'
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        n_iters = len(metric_data[0]) if len(metric_data) > 0 else 0
        title = f'Nested Cross-Validation Results - {metric.upper()}\n({n_iters} iterations with different random seeds)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Rotate x-axis labels if needed
    if len(feature_names) > 5:
        ax.set_xticklabels(feature_names, rotation=45, ha='right')

    # Grid
    ax.grid(axis='y', alpha=0.3)

    # Set y-axis limits
    if metric == 'auc':
        ax.set_ylim(0.5, 1.0)
    else:
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Box plot saved to: {output_path}")

    plt.close()


def plot_results_barplot(
    results_dict: Dict[str, Dict[str, any]],
    metric: str = 'auc',
    title: str = None,
    ylabel: str = None,
    figsize: Tuple[int, int] = (14, 8),
    output_path: str = None,
    dpi: int = 300,
    colors: List[str] = None
):
    """
    Create bar plot with error bars comparing mean performance.

    Args:
        results_dict: Dictionary mapping feature set names to results
        metric: Metric to plot ('auc' or 'accuracy')
        title: Plot title (auto-generated if None)
        ylabel: Y-axis label (auto-generated if None)
        figsize: Figure size
        output_path: Path to save plot (optional)
        dpi: DPI for saved figure
        colors: List of colors for bars
    """
    # Extract data
    feature_names = []
    means = []
    stds = []

    for name, results in results_dict.items():
        feature_names.append(name)
        means.append(results[f'mean_{metric}'])
        stds.append(results[f'std_{metric}'])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Bar positions
    x_pos = np.arange(len(feature_names))

    # Colors
    if colors is None:
        colors = ['steelblue', 'seagreen', 'coral', 'gold', 'orchid',
                  'slateblue', 'orange', 'teal'] * 2

    # Create bars
    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=5,
        color=colors[:len(feature_names)],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.2
    )

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            i, mean + std + 0.01,
            f'{mean:.3f}±{std:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Labels and title
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)

    if ylabel is None:
        ylabel = 'AUC-ROC Score' if metric == 'auc' else 'Accuracy Score'
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f'Performance Comparison - {metric.upper()} (Mean ± Std)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Rotate x-axis labels if needed
    if len(feature_names) > 5:
        ax.set_xticklabels(feature_names, rotation=45, ha='right')

    # Grid
    ax.grid(axis='y', alpha=0.3)

    # Set y-axis limits
    if metric == 'auc':
        ax.set_ylim(0.5, 1.0)
    else:
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Bar plot saved to: {output_path}")

    plt.close()


def plot_model_comparison(
    model_results: Dict[str, Dict[str, Dict[str, any]]],
    metric: str = 'auc',
    figsize: Tuple[int, int] = (16, 10),
    output_dir: str = None,
    dpi: int = 300
):
    """
    Create separate plots for each model comparing feature sets.

    Args:
        model_results: Nested dict {model_name: {feature_set: results}}
        metric: Metric to plot
        figsize: Figure size
        output_dir: Directory to save plots
        dpi: DPI for saved figures
    """
    for model_name, feature_results in model_results.items():
        print(f"\nGenerating plots for {model_name}...")

        # Box plot
        if output_dir:
            boxplot_path = os.path.join(output_dir, f"{model_name}_{metric}_boxplot.png")
        else:
            boxplot_path = None

        plot_results_boxplot(
            results_dict=feature_results,
            metric=metric,
            title=f'{model_name} - {metric.upper()} Comparison',
            figsize=figsize,
            output_path=boxplot_path,
            dpi=dpi
        )

        # Bar plot
        if output_dir:
            barplot_path = os.path.join(output_dir, f"{model_name}_{metric}_barplot.png")
        else:
            barplot_path = None

        plot_results_barplot(
            results_dict=feature_results,
            metric=metric,
            title=f'{model_name} - {metric.upper()} Performance',
            figsize=figsize,
            output_path=barplot_path,
            dpi=dpi
        )


def save_results_table(
    results_dict: Dict[str, Dict[str, any]],
    output_path: str,
    metrics: List[str] = ['auc', 'accuracy']
):
    """
    Save results summary table as CSV.

    Args:
        results_dict: Dictionary mapping feature set names to results
        output_path: Path to save CSV
        metrics: List of metrics to include
    """
    rows = []

    for feature_name, results in results_dict.items():
        row = {'feature_set': feature_name}

        for metric in metrics:
            row[f'{metric}_mean'] = results[f'mean_{metric}']
            row[f'{metric}_std'] = results[f'std_{metric}']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by primary metric (first one)
    primary_metric = metrics[0]
    df = df.sort_values(f'{primary_metric}_mean', ascending=False)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nResults table saved to: {output_path}")
    print(df.to_string(index=False))


def save_detailed_results(
    results_dict: Dict[str, Dict[str, any]],
    output_path: str
):
    """
    Save detailed results including all scores per iteration.

    For NLP evaluations, also saves the selected embedding variant per fold.

    Args:
        results_dict: Dictionary mapping feature set names to results
        output_path: Path to save CSV
    """
    data = []

    for feature_name, results in results_dict.items():
        # Handle both standard format and model selection format
        auc_scores = results.get('auc_scores') or results.get('all_auc_scores', [])
        accuracy_scores = results.get('accuracy_scores') or results.get('all_accuracy_scores', [])
        best_variants = results.get('best_variants', None)
        selected_models = results.get('selected_models', None)  # For model selection results

        # Build data rows with available information
        if best_variants is not None or selected_models is not None:
            # NLP evaluation or model selection - include additional info
            n_iterations = len(auc_scores)

            for i, (auc, acc) in enumerate(zip(auc_scores, accuracy_scores)):
                row = {
                    'feature_set': feature_name,
                    'iteration': i + 1,
                    'auc': auc,
                    'accuracy': acc
                }

                # Add variant info if available (NLP evaluation)
                if best_variants is not None:
                    variants_per_iter = len(best_variants) // n_iterations if n_iterations > 0 else 0
                    start_idx = i * variants_per_iter
                    end_idx = start_idx + variants_per_iter
                    iter_variants = best_variants[start_idx:end_idx]
                    row['selected_variants'] = ', '.join(iter_variants) if iter_variants else None

                # Add model info if available (model selection)
                if selected_models is not None:
                    # For model selection, selected_models has one entry per fold
                    # Map to iterations (assuming n_outer folds per iteration)
                    # This is a simplified mapping - may need adjustment based on actual structure
                    row['selected_model'] = selected_models[i] if i < len(selected_models) else None

                data.append(row)
        else:
            # Standard evaluation (no variants or model selection)
            for i, (auc, acc) in enumerate(zip(auc_scores, accuracy_scores)):
                data.append({
                    'feature_set': feature_name,
                    'iteration': i + 1,
                    'auc': auc,
                    'accuracy': acc
                })

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Detailed results saved to: {output_path}")


def save_comprehensive_results(
    results_dict: Dict[str, Dict[str, any]],
    output_dir: str,
    model_name: str
):
    """
    Save comprehensive evaluation results for reproducibility.

    Saves three files:
    1. {model_name}_scores.csv - All iteration scores (for reproducing box plots)
    2. {model_name}_summary_stats.csv - Mean/std statistics
    3. {model_name}_config.csv - Best hyperparameters per iteration (if available)

    Args:
        results_dict: Dictionary mapping feature set names to results
        output_dir: Directory to save files
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save all iteration scores (for box plots)
    scores_data = []
    for feature_name, results in results_dict.items():
        auc_scores = results.get('auc_scores', [])
        accuracy_scores = results.get('accuracy_scores', [])
        seeds = results.get('seeds', list(range(len(auc_scores))))

        for i, (auc, acc, seed) in enumerate(zip(auc_scores, accuracy_scores, seeds)):
            scores_data.append({
                'feature_set': feature_name,
                'iteration': i + 1,
                'seed': seed,
                'auc': auc,
                'accuracy': acc
            })

    scores_df = pd.DataFrame(scores_data)
    scores_path = os.path.join(output_dir, f"{model_name}_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"  Saved iteration scores to: {scores_path}")

    # 2. Save summary statistics
    summary_data = []
    for feature_name, results in results_dict.items():
        summary_data.append({
            'feature_set': feature_name,
            'mean_auc': results.get('mean_auc', np.nan),
            'std_auc': results.get('std_auc', np.nan),
            'mean_accuracy': results.get('mean_accuracy', np.nan),
            'std_accuracy': results.get('std_accuracy', np.nan),
            'n_iterations': len(results.get('auc_scores', []))
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"{model_name}_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary statistics to: {summary_path}")

    # 3. Save feature importance if available
    importance_saved = False
    for feature_name, results in results_dict.items():
        if 'feature_importance' in results and results['feature_importance'] is not None:
            importance_df = results['feature_importance'].copy()
            importance_df.insert(0, 'feature_set', feature_name)
            importance_path = os.path.join(output_dir, f"{model_name}_{feature_name}_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            if not importance_saved:
                print(f"  Saved feature importances to: {output_dir}/{model_name}_*_importance.csv")
                importance_saved = True


def save_permutation_results(
    perm_results: Dict,
    output_dir: str,
    model_name: str
):
    """
    Save permutation test results as CSV and plots.

    Args:
        perm_results: Dict mapping test pairs to result dicts
        output_dir: Directory to save results
        model_name: Name of the model
    """
    if not perm_results:
        print("  No permutation test results to save")
        return

    # 1. Save summary CSV
    summary_rows = []
    for test_name, result in perm_results.items():
        summary_rows.append({
            'test': test_name,
            'feature_set_1': result['feature_set_1'],
            'feature_set_2': result['feature_set_2'],
            'auc_1': result['auc_1'],
            'auc_2': result['auc_2'],
            'delta_obs': result['delta_obs'],
            'p_value': result['p_value'],
            'significant': result['significant'],
            'n_iterations': result['n_iterations'],
            'n_folds': result['n_folds'],
            'n_permutations': result['n_permutations']
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"{model_name}_permutation_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  ✓ Saved permutation summary: {summary_path}")

    # 1.5. Save full results including delta_perm for replotting
    import pickle
    full_results_path = os.path.join(output_dir, f"{model_name}_permutation_full_results.pkl")
    with open(full_results_path, 'wb') as f:
        pickle.dump(perm_results, f)
    print(f"  ✓ Saved full permutation results (for replotting): {full_results_path}")

    # 2. Save null distribution plots
    n_tests = len(perm_results)
    fig, axes = plt.subplots(n_tests, 1, figsize=(10, 4*n_tests))
    if n_tests == 1:
        axes = [axes]

    for idx, (test_name, result) in enumerate(perm_results.items()):
        ax = axes[idx]

        # Plot null distribution
        ax.hist(result['delta_perm'], bins=50, alpha=0.7,
                color='lightblue', edgecolor='black', label='Null distribution')

        # Mark observed difference
        ax.axvline(result['delta_obs'], color='red', linestyle='--',
                   linewidth=2, label=f"Observed: {result['delta_obs']:.4f}")

        # Mark mean of null distribution
        ax.axvline(np.mean(result['delta_perm']), color='gray', linestyle=':',
                   linewidth=1, alpha=0.7, label=f"Null mean: {np.mean(result['delta_perm']):.4f}")

        # Add p-value annotation
        significance_text = '✓ Significant' if result['significant'] else 'Not significant'
        ax.text(0.95, 0.95, f"p = {result['p_value']:.4f}\n{significance_text}",
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        ax.set_xlabel('AUC Difference (Macro-Averaged)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{test_name}\n{result["feature_set_2"]} vs {result["feature_set_1"]}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_permutation_distributions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved null distribution plots: {plot_path}")
