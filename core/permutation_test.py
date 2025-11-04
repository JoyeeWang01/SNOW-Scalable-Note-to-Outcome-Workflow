"""
Permutation test for comparing predictive power of two feature sets.

This module implements cell-level permutation testing with macro-averaged AUC differences:
- For each iteration r and fold k, swap predictions with 50% probability
- Compute AUC difference per cell: Δ^(r,k) = AUC(p̂₂) - AUC(p̂₁)
- Macro-average: Δ = (1/RK) Σ Δ^(r,k)
- p-value with continuity correction: (1 + count) / (B + 1)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple
from core.log_utils import print


def compute_cell_differences(
    predictions_1: Dict,
    predictions_2: Dict
) -> Tuple[float, np.ndarray]:
    """
    Compute per-cell AUC differences and macro-average.

    Args:
        predictions_1: Nested dict {iteration_i: {fold_k: {y_true, y_pred}}}
        predictions_2: Nested dict {iteration_i: {fold_k: {y_true, y_pred}}}

    Returns:
        Tuple of (delta_obs, cell_deltas):
            - delta_obs: Macro-averaged difference across all cells
            - cell_deltas: Array of shape (R, K) with per-cell differences
    """
    # Get dimensions
    R = len(predictions_1)  # number of iterations
    K = len(predictions_1['iteration_0'])  # number of folds

    cell_deltas = np.zeros((R, K))

    for r in range(R):
        for k in range(K):
            # Extract predictions for this cell
            cell_1 = predictions_1[f'iteration_{r}'][k]
            cell_2 = predictions_2[f'iteration_{r}'][k]

            y_true = cell_1['y_true']
            p1 = cell_1['y_pred']
            p2 = cell_2['y_pred']

            # Compute AUC for each model
            auc1 = roc_auc_score(y_true, p1)
            auc2 = roc_auc_score(y_true, p2)

            # Store difference
            cell_deltas[r, k] = auc2 - auc1

    # Macro-average across all cells
    delta_obs = np.mean(cell_deltas)

    return delta_obs, cell_deltas


def permutation_test_cell_level(
    predictions_1: Dict,
    predictions_2: Dict,
    n_permutations: int = 10000,
    random_seed: int = 42,
    verbosity: int = 1
) -> Tuple[float, float, np.ndarray]:
    """
    Perform cell-level permutation test with macro-averaged differences.

    Under H₀: both models are equally good, so swapping predictions within
    each cell should not affect the macro-averaged difference.

    Args:
        predictions_1: Nested dict {iteration_i: {fold_k: {y_true, y_pred}}}
        predictions_2: Nested dict {iteration_i: {fold_k: {y_true, y_pred}}}
        n_permutations: Number of permutation iterations (B)
        random_seed: Random seed for reproducibility
        verbosity: Verbosity level

    Returns:
        Tuple of (delta_obs, p_value, delta_perm):
            - delta_obs: Observed macro-averaged difference
            - p_value: One-sided p-value with continuity correction
            - delta_perm: Array of permuted differences (length B)
    """
    if verbosity >= 1:
        print(f"\nComputing observed statistic...")

    # Step 1: Compute observed statistic
    delta_obs, _ = compute_cell_differences(predictions_1, predictions_2)

    if verbosity >= 1:
        print(f"  Observed macro-averaged difference: {delta_obs:.4f}")
        print(f"\nRunning {n_permutations} permutations...")

    # Step 2: Extract structure
    R = len(predictions_1)  # iterations
    K = len(predictions_1['iteration_0'])  # folds

    # Step 3: Permutation loop
    delta_perm = np.zeros(n_permutations)
    rng = np.random.default_rng(random_seed)

    for b in range(n_permutations):
        cell_deltas_b = np.zeros((R, K))

        # For each cell, randomly swap predictions
        for r in range(R):
            for k in range(K):
                # Get predictions for this cell
                cell_1 = predictions_1[f'iteration_{r}'][k]
                cell_2 = predictions_2[f'iteration_{r}'][k]

                y_true = cell_1['y_true']
                p1 = cell_1['y_pred']
                p2 = cell_2['y_pred']

                # Generate swap mask for this cell
                n_samples = len(y_true)
                swap_mask = rng.random(n_samples) < 0.5

                # Swap predictions per observation
                p1_perm = np.where(swap_mask, p2, p1)
                p2_perm = np.where(swap_mask, p1, p2)

                # Compute permuted cell difference
                auc1_perm = roc_auc_score(y_true, p1_perm)
                auc2_perm = roc_auc_score(y_true, p2_perm)
                cell_deltas_b[r, k] = auc2_perm - auc1_perm

        # Macro-average across all cells
        delta_perm[b] = np.mean(cell_deltas_b)

        # Progress reporting
        if verbosity >= 2 and (b + 1) % 1000 == 0:
            print(f"  Completed {b + 1}/{n_permutations} permutations")

    # Step 4: Compute p-value with continuity correction
    # One-sided test: H₀: model 2 is not better than model 1
    p_value = (1 + np.sum(delta_perm >= delta_obs)) / (n_permutations + 1)

    if verbosity >= 1:
        print(f"\nPermutation test results:")
        print(f"  Observed difference: {delta_obs:.4f}")
        print(f"  Mean permuted difference: {np.mean(delta_perm):.4f}")
        print(f"  Std permuted difference: {np.std(delta_perm):.4f}")
        print(f"  p-value (one-sided): {p_value:.4f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

    return delta_obs, p_value, delta_perm


def run_permutation_test_pipeline(
    X1: np.ndarray,
    X2: np.ndarray,
    y: np.ndarray,
    model1,
    model2,
    param_grid1: Dict,
    param_grid2: Dict,
    n_iterations: int,
    master_seed: int,
    n_outer: int,
    n_inner: int,
    n_inner_repeats: int,
    imputation_method: str,
    imputation_params: Dict,
    feature_set_1_name: str,
    feature_set_2_name: str,
    n_permutations: int = 10000,
    perm_seed: int = 42,
    verbosity: int = 1,
    predictions_1: Dict = None,
    predictions_2: Dict = None,
    auc_stats_1: Dict = None,
    auc_stats_2: Dict = None
) -> Dict:
    """
    Complete permutation test pipeline.

    Steps:
    1. Run nested CV for both models with same seeds (same folds) OR use pre-computed predictions
    2. Collect predictions from all iterations and folds
    3. Run cell-level permutation test
    4. Return comprehensive results

    Args:
        X1: Feature matrix for model 1
        X2: Feature matrix for model 2
        y: Target labels
        model1: Model instance for feature set 1
        model2: Model instance for feature set 2
        param_grid1: Hyperparameter grid for model 1
        param_grid2: Hyperparameter grid for model 2
        n_iterations: Number of nested CV iterations
        master_seed: Master random seed (ensures same folds for both models)
        n_outer: Number of outer CV folds
        n_inner: Number of inner CV folds
        n_inner_repeats: Number of inner CV repeats
        imputation_method: Imputation method
        imputation_params: Imputation parameters
        feature_set_1_name: Name of feature set 1
        feature_set_2_name: Name of feature set 2
        n_permutations: Number of permutation iterations
        perm_seed: Random seed for permutation test
        verbosity: Verbosity level
        predictions_1: Pre-computed predictions for feature set 1 (optional, to avoid re-running CV)
        predictions_2: Pre-computed predictions for feature set 2 (optional, to avoid re-running CV)
        auc_stats_1: AUC statistics for feature set 1 (dict with 'mean_auc', 'std_auc')
        auc_stats_2: AUC statistics for feature set 2 (dict with 'mean_auc', 'std_auc')

    Returns:
        Dictionary with results:
            - 'feature_set_1': Name of feature set 1
            - 'feature_set_2': Name of feature set 2
            - 'auc_1': Mean AUC of model 1
            - 'auc_2': Mean AUC of model 2
            - 'delta_obs': Observed macro-averaged difference
            - 'p_value': One-sided p-value
            - 'significant': Boolean (p < 0.05)
            - 'delta_perm': Array of permuted differences
            - 'n_iterations': Number of iterations
            - 'n_folds': Number of folds
            - 'n_permutations': Number of permutations
    """
    from core.evaluation import run_multiple_evaluations

    print(f"\n{'='*80}")
    print(f"PERMUTATION TEST: {feature_set_2_name} vs {feature_set_1_name}")
    print(f"{'='*80}")

    # Check if we can use pre-computed predictions
    if predictions_1 is not None and predictions_2 is not None:
        if verbosity >= 1:
            print(f"Using pre-computed predictions from main evaluation (skipping CV re-run)")
        results1 = {
            'iteration_predictions': predictions_1,
            'mean_auc': auc_stats_1.get('mean_auc', 0) if auc_stats_1 else 0,
            'std_auc': auc_stats_1.get('std_auc', 0) if auc_stats_1 else 0
        }
        results2 = {
            'iteration_predictions': predictions_2,
            'mean_auc': auc_stats_2.get('mean_auc', 0) if auc_stats_2 else 0,
            'std_auc': auc_stats_2.get('std_auc', 0) if auc_stats_2 else 0
        }
    else:
        # Need to run CV to get predictions
        if verbosity >= 1:
            print(f"Running nested CV with {n_iterations} iterations and {n_outer} folds...")

        # Step 1: Run nested CV for feature set 1 with prediction collection
        print(f"\nEvaluating {feature_set_1_name}...")
        results1 = run_multiple_evaluations(
            X=X1,
            y=y,
            model=model1,
            param_grid=param_grid1,
            n_iterations=n_iterations,
            master_seed=master_seed,
            n_outer=n_outer,
            n_inner=n_inner,
            n_inner_repeats=n_inner_repeats,
            imputation_method=imputation_method,
            imputation_params=imputation_params,
            collect_predictions=True,
            verbosity=verbosity
        )

        # Step 2: Run nested CV for feature set 2 (SAME MASTER SEED → SAME FOLDS)
        print(f"\nEvaluating {feature_set_2_name}...")
        results2 = run_multiple_evaluations(
            X=X2,
            y=y,
            model=model2,
            param_grid=param_grid2,
            n_iterations=n_iterations,
            master_seed=master_seed,  # CRITICAL: same seed ensures same folds
            n_outer=n_outer,
            n_inner=n_inner,
            n_inner_repeats=n_inner_repeats,
            imputation_method=imputation_method,
            imputation_params=imputation_params,
            collect_predictions=True,
            verbosity=verbosity
        )

    # Step 3: Run permutation test
    delta_obs, p_value, delta_perm = permutation_test_cell_level(
        predictions_1=results1['iteration_predictions'],
        predictions_2=results2['iteration_predictions'],
        n_permutations=n_permutations,
        random_seed=perm_seed,
        verbosity=verbosity
    )

    # Step 4: Compile results
    test_results = {
        'feature_set_1': feature_set_1_name,
        'feature_set_2': feature_set_2_name,
        'auc_1': results1['mean_auc'],
        'auc_2': results2['mean_auc'],
        'delta_obs': delta_obs,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'delta_perm': delta_perm,
        'n_iterations': n_iterations,
        'n_folds': n_outer,
        'n_permutations': n_permutations
    }

    print(f"\n{'='*80}")
    print(f"PERMUTATION TEST RESULTS")
    print(f"{'='*80}")
    print(f"\nMean AUC from nested CV evaluation:")
    print(f"  {feature_set_2_name}: {results2['mean_auc']:.4f} (± {results2.get('std_auc', 0):.4f})")
    print(f"  {feature_set_1_name}: {results1['mean_auc']:.4f} (± {results1.get('std_auc', 0):.4f})")
    print(f"\nObserved difference (macro-averaged across all folds):")
    print(f"  Δ_obs = {delta_obs:.4f}")
    print(f"\nNull distribution from {n_permutations} permutations:")
    print(f"  Mean: {np.mean(delta_perm):.4f}")
    print(f"  Std: {np.std(delta_perm):.4f}")
    print(f"  95% CI: [{np.percentile(delta_perm, 2.5):.4f}, {np.percentile(delta_perm, 97.5):.4f}]")
    print(f"\nStatistical test:")
    print(f"  p-value (one-sided): {p_value:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"\nConclusion:")
    print(f"  {feature_set_2_name} {'SIGNIFICANTLY' if p_value < 0.05 else 'does NOT significantly'} "
          f"outperform {feature_set_1_name}")
    print(f"{'='*80}\n")

    return test_results
