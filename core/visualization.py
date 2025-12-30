"""
Visualization functions for evaluation results.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc
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
            # Create safe filename from feature set name
            safe_name = feature_name.replace(' ', '_').replace('+', '').replace('/', '_')
            importance_path = os.path.join(output_dir, f"{model_name}_{safe_name}_feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            if not importance_saved:
                print(f"  Saved feature importances to: {output_dir}/{model_name}_*_feature_importance.csv")
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


def save_roc_data(
    iteration_predictions: Dict,
    output_path: str
):
    """
    Save raw predictions to CSV for ROC plot regeneration.

    Args:
        iteration_predictions: Nested dict with predictions per iteration and fold
            Format: {iteration_key: {fold_idx: {'y_true': array, 'y_pred': array, 'test_idx': array}}}
        output_path: Path to save CSV file

    Saves CSV with columns: iteration, fold, sample_idx, y_true, y_pred
    """
    rows = []

    for iter_key, fold_preds in iteration_predictions.items():
        # Extract iteration number from key (e.g., 'iteration_0' -> 0)
        iter_num = int(iter_key.split('_')[1])

        for fold_idx, fold_data in fold_preds.items():
            y_true = fold_data['y_true']
            y_pred = fold_data['y_pred']
            test_idx = fold_data.get('test_idx', np.arange(len(y_true)))

            for i, (sample_idx, true_val, pred_val) in enumerate(zip(test_idx, y_true, y_pred)):
                rows.append({
                    'iteration': iter_num,
                    'fold': fold_idx,
                    'sample_idx': int(sample_idx),
                    'y_true': int(true_val),
                    'y_pred': float(pred_val)
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw predictions saved to: {output_path}")
    print(f"  Total predictions: {len(df)}")
    print(f"  Iterations: {df['iteration'].nunique()}, Folds: {df['fold'].nunique()}")
    print(f"  Samples: {df['sample_idx'].nunique()}")


def plot_pooled_roc_curve(
    iteration_predictions: Dict,
    title: str = None,
    color: str = '#2E86AB',
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 300
):
    """
    Plot pooled ROC curve (all predictions concatenated into one curve).

    Args:
        iteration_predictions: Dictionary from results['iteration_predictions']
                              Format: {iteration_X: {fold_Y: {y_true, y_pred, test_idx}}}
        title: Plot title
        color: Line color
        output_path: Path to save plot (optional)
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        fig, ax, roc_data: Figure, axes, and ROC curve data dict
    """
    # Collect ALL predictions from all iterations and folds
    all_y_true = []
    all_y_pred = []

    for iter_key, fold_dict in iteration_predictions.items():
        for fold_key, fold_data in fold_dict.items():
            all_y_true.extend(fold_data['y_true'])
            all_y_pred.extend(fold_data['y_pred'])

    # Compute single ROC curve from pooled data
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred)
    roc_auc = auc(fpr, tpr)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, color=color,
            label=f'ROC curve (AUC = {roc_auc:.3f})')

    # Plot diagonal (chance)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance (AUC = 0.500)')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Pooled ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved pooled ROC curve: {output_path}")

    # Return data for regeneration
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'n_samples': len(all_y_true)
    }

    return fig, ax, roc_data


def plot_mean_roc_curve(
    iteration_predictions: Dict,
    title: str = None,
    color: str = '#2E86AB',
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 300,
    show_individual: bool = False,
    individual_alpha: float = 0.1
):
    """
    Plot mean ROC curve with confidence bands (averaged across iterations).

    Args:
        iteration_predictions: Dictionary from results['iteration_predictions']
        title: Plot title
        color: Line color
        output_path: Path to save plot (optional)
        figsize: Figure size
        dpi: DPI for saved figure
        show_individual: If True, plot individual iteration ROCs (faded)
        individual_alpha: Alpha for individual curves

    Returns:
        fig, ax, roc_data: Figure, axes, and ROC curve data dict
    """
    # Compute ROC for each iteration (pooling folds within iteration)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=figsize)

    for iter_key, fold_dict in iteration_predictions.items():
        # Pool predictions within this iteration
        iter_y_true = []
        iter_y_pred = []
        for fold_data in fold_dict.values():
            iter_y_true.extend(fold_data['y_true'])
            iter_y_pred.extend(fold_data['y_pred'])

        # Compute ROC for this iteration
        fpr, tpr, _ = roc_curve(iter_y_true, iter_y_pred)

        # Interpolate to fixed FPR values
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Plot individual curve if requested
        if show_individual:
            ax.plot(fpr, tpr, alpha=individual_alpha, color=color, linewidth=1)

    # Calculate mean and std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Plot mean ROC
    ax.plot(mean_fpr, mean_tpr, linewidth=2, color=color,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    # Add confidence band (mean ± 1 std)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                     alpha=0.2, color=color, label='± 1 std. dev.')

    # Plot diagonal (chance)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance (AUC = 0.500)')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Mean ROC Curve with Confidence Bands', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved mean ROC curve: {output_path}")

    # Return data for regeneration
    roc_data = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'aucs': aucs,
        'n_iterations': len(iteration_predictions)
    }

    return fig, ax, roc_data


def plot_roc_curves_comparison(
    results_dict: Dict[str, Dict],
    output_dir: str,
    plot_type: str = 'both',
    feature_set_name: str = None,
    model_name: str = None,
    colors: Dict[str, str] = None,
    dpi: int = 300
):
    """
    Plot ROC curves for multiple models or feature sets.

    Args:
        results_dict: Dict mapping names to results containing 'iteration_predictions'
        output_dir: Directory to save plots
        plot_type: 'pooled', 'mean', or 'both'
        feature_set_name: Name for the feature set (for filenames)
        model_name: Name for the model (for filenames)
        colors: Dict mapping names to colors (auto-generated if None)
        dpi: DPI for saved figures

    Returns:
        Dict with 'pooled' and/or 'mean' keys containing ROC data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Auto-generate colors if not provided
    if colors is None:
        default_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
                          '#8B4789', '#D4B483', '#457B9D', '#E63946', '#06A77D']
        colors = {name: default_colors[i % len(default_colors)]
                  for i, name in enumerate(results_dict.keys())}

    all_roc_data = {}

    # Generate pooled ROC if requested
    if plot_type in ['pooled', 'both']:
        print(f"  Generating pooled ROC curves...")

        for name, results in results_dict.items():
            if 'iteration_predictions' not in results:
                print(f"    Warning: No iteration_predictions for {name}")
                continue

            color = colors.get(name, '#2E86AB')
            title = f'{model_name} - {name} (Pooled)' if model_name else f'{name} (Pooled)'

            safe_name = name.replace(' ', '_').replace('+', '').replace('/', '_')
            filename = f"{model_name}_{safe_name}_roc_pooled.png" if model_name else f"{safe_name}_roc_pooled.png"
            output_path = os.path.join(output_dir, filename)

            fig, ax, roc_data = plot_pooled_roc_curve(
                iteration_predictions=results['iteration_predictions'],
                title=title,
                color=color,
                output_path=output_path,
                dpi=dpi
            )
            plt.close(fig)

            if 'pooled' not in all_roc_data:
                all_roc_data['pooled'] = {}
            all_roc_data['pooled'][name] = roc_data

    # Generate mean ROC if requested
    if plot_type in ['mean', 'both']:
        print(f"  Generating mean ROC curves...")

        for name, results in results_dict.items():
            if 'iteration_predictions' not in results:
                continue

            color = colors.get(name, '#2E86AB')
            title = f'{model_name} - {name} (Mean ± Std)' if model_name else f'{name} (Mean ± Std)'

            safe_name = name.replace(' ', '_').replace('+', '').replace('/', '_')
            filename = f"{model_name}_{safe_name}_roc_mean.png" if model_name else f"{safe_name}_roc_mean.png"
            output_path = os.path.join(output_dir, filename)

            fig, ax, roc_data = plot_mean_roc_curve(
                iteration_predictions=results['iteration_predictions'],
                title=title,
                color=color,
                output_path=output_path,
                dpi=dpi
            )
            plt.close(fig)

            if 'mean' not in all_roc_data:
                all_roc_data['mean'] = {}
            all_roc_data['mean'][name] = roc_data

    return all_roc_data


def plot_roc_curves_combined(
    results_dict: Dict[str, Dict[str, any]],
    output_dir: str,
    model_name: str,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 300
):
    """
    Generate and save both pooled and mean ROC plots for all feature sets.

    This is a convenience function that:
    1. Saves raw predictions to CSV for regeneration
    2. Generates pooled and mean ROC plots for each feature set

    Args:
        results_dict: Dictionary mapping feature set names to results
        output_dir: Directory to save plots and data
        model_name: Name of the model (for filenames)
        figsize: Figure size
        dpi: DPI for saved figures
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating ROC curves for {model_name}...")

    # Save raw predictions for each feature set
    for feature_name, results in results_dict.items():
        if 'iteration_predictions' not in results:
            print(f"  Warning: No iteration_predictions for {feature_name}, skipping")
            continue

        # Create safe filename
        safe_name = feature_name.replace(' ', '_').replace('+', '').replace('/', '_')
        pred_path = os.path.join(output_dir, f"{model_name}_{safe_name}_predictions.csv")

        save_roc_data(
            iteration_predictions=results['iteration_predictions'],
            output_path=pred_path
        )

    # Generate ROC plots for all feature sets
    plot_roc_curves_comparison(
        results_dict=results_dict,
        output_dir=output_dir,
        plot_type='both',
        model_name=model_name,
        dpi=dpi
    )

    print(f"  ✓ ROC curves generated for {model_name}")
