"""
Nested cross-validation functions for model evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, Any, Tuple
from core.log_utils import print

from core.ml_models import SVDImputer


def get_imputer(method: str = 'mean', random_state: int = None, **kwargs):
    """
    Get imputer based on method name.

    Args:
        method: Imputation method ('mean', 'mice', 'svd', 'knn')
        random_state: Random seed
        **kwargs: Additional parameters for specific imputers

    Returns:
        Scikit-learn compatible imputer
    """
    if method == 'mean':
        return SimpleImputer(strategy='mean')
    elif method == 'mice':
        return IterativeImputer(
            max_iter=kwargs.get('max_iter', 10),
            random_state=random_state
        )
    elif method == 'svd':
        return SVDImputer(
            n_components=kwargs.get('n_components', 3),
            max_iter=kwargs.get('max_iter', 500),
            tol=kwargs.get('tol', 1e-4),
            random_state=random_state
        )
    elif method == 'knn':
        return KNNImputer(n_neighbors=kwargs.get('n_neighbors', 5))
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def nested_cv_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    model,
    param_grid: Dict[str, Any],
    n_outer: int = 3,
    n_inner: int = 3,
    n_inner_repeats: int = 1,
    imputation_method: str = 'mean',
    random_seed: int = None,
    imputation_params: Dict[str, Any] = None,
    verbosity: int = 0,
    collect_predictions: bool = False
) -> Dict[str, Any]:
    """
    Perform nested cross-validation for model evaluation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        model: Scikit-learn compatible model
        param_grid: Hyperparameter grid for GridSearchCV
        n_outer: Number of outer CV folds
        n_inner: Number of inner CV folds
        imputation_method: Method for missing value imputation
        random_seed: Random seed for reproducibility
        imputation_params: Additional parameters for imputer
        verbosity: Verbosity level (0=silent, 1=progress, 2=detailed)

    Returns:
        Dictionary with results:
            - auc_scores: List of AUC scores per fold
            - accuracy_scores: List of accuracy scores per fold
            - mean_auc: Mean AUC across folds
            - std_auc: Standard deviation of AUC
            - mean_accuracy: Mean accuracy across folds
            - std_accuracy: Standard deviation of accuracy
            - best_params: Best hyperparameters per fold
    """
    if imputation_params is None:
        imputation_params = {}

    # Initialize result storage
    auc_scores = []
    accuracy_scores = []
    best_params_list = []
    feature_importances = []  # Track feature importance per fold

    # Initialize prediction storage if requested
    if collect_predictions:
        fold_predictions = {}

    # Create outer CV splitter
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_seed)

    # Create RNG for generating fold-specific seeds
    fold_rng = np.random.RandomState(random_seed)

    # Outer loop
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        if verbosity >= 1:
            print(f"  Outer fold {fold_idx + 1}/{n_outer}")

        # Generate fold-specific seed and derive component-specific seeds
        fold_seed = fold_rng.randint(0, 100000)
        component_rng = np.random.RandomState(fold_seed)
        inner_cv_seed = component_rng.randint(0, 100000)
        model_seed = component_rng.randint(0, 100000)

        # Derive imputer seed from model seed
        imputer_rng = np.random.RandomState(model_seed)
        imputer_seed = imputer_rng.randint(0, 100000)

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Imputation
        imputer = get_imputer(imputation_method, random_state=imputer_seed, **imputation_params)
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # Inner CV for hyperparameter tuning
        if n_inner_repeats > 1:
            inner_cv = RepeatedStratifiedKFold(
                n_splits=n_inner,
                n_repeats=n_inner_repeats,
                random_state=inner_cv_seed
            )
        else:
            inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=inner_cv_seed)

        # Set model random_state if the model has this parameter
        from sklearn.base import clone
        model_clone = clone(model)
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = model_seed

        grid_search = GridSearchCV(
            estimator=model_clone,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train_scaled, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        best_params_list.append(grid_search.best_params_)

        if verbosity >= 1:
            # Format params nicely
            params_str = ", ".join([f"{k}={v}" for k, v in grid_search.best_params_.items()])
            print(f"    Selected params: {params_str}")

        # Predict on test fold
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = best_model.decision_function(X_test_scaled)

        y_pred = best_model.predict(X_test_scaled)

        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        auc_scores.append(auc)
        accuracy_scores.append(accuracy)

        if verbosity >= 1:
            print(f"    Fold AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

        # Store predictions if requested
        if collect_predictions:
            fold_predictions[fold_idx] = {
                'y_true': y_test.copy(),
                'y_pred': y_pred_proba.copy(),
                'test_idx': test_idx.copy()
            }

        # Extract feature importance if available
        fold_importance = None
        if hasattr(best_model, 'feature_importances_'):
            # Tree-based models
            fold_importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            # Linear models (use absolute value)
            fold_importance = np.abs(best_model.coef_[0])

        if fold_importance is not None:
            feature_importances.append(fold_importance)

        if verbosity >= 2:
            print(f"    AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    # Aggregate results
    results = {
        'auc_scores': auc_scores,
        'accuracy_scores': accuracy_scores,
        'mean_auc': np.mean(auc_scores),
        'std_auc': np.std(auc_scores),
        'mean_accuracy': np.mean(accuracy_scores),
        'std_accuracy': np.std(accuracy_scores),
        'best_params': best_params_list
    }

    # Add aggregated feature importance if available
    if feature_importances:
        # Average importance across folds
        mean_importance = np.mean(feature_importances, axis=0)
        results['feature_importance'] = mean_importance

    # Add fold predictions if requested
    if collect_predictions:
        results['fold_predictions'] = fold_predictions

    if verbosity >= 1:
        print(f"  Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

    return results


def run_multiple_evaluations(
    X: np.ndarray,
    y: np.ndarray,
    model,
    param_grid: Dict[str, Any],
    n_iterations: int = 100,
    master_seed: int = 42,
    n_outer: int = 3,
    n_inner: int = 3,
    n_inner_repeats: int = 1,
    imputation_method: str = 'mean',
    imputation_params: Dict[str, Any] = None,
    verbosity: int = 1,
    feature_names: list = None,
    collect_predictions: bool = False
) -> Dict[str, Any]:
    """
    Run nested CV evaluation multiple times with different random seeds.

    Args:
        X: Feature matrix
        y: Target labels
        model: Scikit-learn compatible model
        param_grid: Hyperparameter grid
        n_iterations: Number of evaluation iterations
        master_seed: Master seed for generating iteration seeds
        n_outer: Number of outer CV folds
        n_inner: Number of inner CV folds
        imputation_method: Imputation method
        imputation_params: Imputer parameters
        verbosity: Verbosity level

    Returns:
        Dictionary with aggregated results across all iterations
    """
    # Generate seeds for each iteration
    rng = np.random.RandomState(master_seed)
    seeds = rng.randint(0, 10000, size=n_iterations)

    # Storage for results
    all_auc_scores = []
    all_accuracy_scores = []
    all_feature_importances = []

    # Initialize iteration predictions storage if requested
    if collect_predictions:
        iteration_predictions = {}

    # Run evaluations
    for i, seed in enumerate(seeds):
        if verbosity >= 1:
            print(f"\nIteration {i + 1}/{n_iterations} (seed={seed})")

        results = nested_cv_evaluation(
            X=X,
            y=y,
            model=model,
            param_grid=param_grid,
            n_outer=n_outer,
            n_inner=n_inner,
            n_inner_repeats=n_inner_repeats,
            imputation_method=imputation_method,
            random_seed=seed,
            imputation_params=imputation_params,
            verbosity=verbosity,
            collect_predictions=collect_predictions
        )

        all_auc_scores.append(results['mean_auc'])
        all_accuracy_scores.append(results['mean_accuracy'])

        # Collect feature importance if available
        if 'feature_importance' in results:
            all_feature_importances.append(results['feature_importance'])

        # Collect fold predictions if requested
        if collect_predictions and 'fold_predictions' in results:
            iteration_predictions[f'iteration_{i}'] = results['fold_predictions']

    # Aggregate results
    aggregated_results = {
        'auc_scores': all_auc_scores,
        'accuracy_scores': all_accuracy_scores,
        'mean_auc': np.mean(all_auc_scores),
        'std_auc': np.std(all_auc_scores),
        'mean_accuracy': np.mean(all_accuracy_scores),
        'std_accuracy': np.std(all_accuracy_scores),
        'seeds': seeds.tolist()
    }

    # Add feature importance if available and feature names provided
    if all_feature_importances and feature_names is not None:
        # Average importance across all iterations
        mean_importance = np.mean(all_feature_importances, axis=0)
        aggregated_results['feature_importance'] = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_importance
        }).sort_values('importance', ascending=False)

    # Add iteration predictions if requested
    if collect_predictions:
        aggregated_results['iteration_predictions'] = iteration_predictions

    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"AGGREGATED RESULTS ({n_iterations} iterations)")
        print(f"{'='*80}")
        print(f"AUC: {aggregated_results['mean_auc']:.4f} ± {aggregated_results['std_auc']:.4f}")
        print(f"Accuracy: {aggregated_results['mean_accuracy']:.4f} ± {aggregated_results['std_accuracy']:.4f}")

    return aggregated_results
