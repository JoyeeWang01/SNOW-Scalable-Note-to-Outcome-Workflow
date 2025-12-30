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
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)
from typing import Dict, Any, Tuple, List, Iterable
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from core.log_utils import print

from core.ml_models import SVDImputer

SUPPORTED_METRICS = {'auc', 'accuracy', 'aupr', 'f1'}


class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to treat decision threshold as a tunable hyperparameter for models with predict_proba.
    Exposes base estimator params via estimator__* so GridSearchCV can tune both model params and threshold together.
    """
    def __init__(self, estimator, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def get_params(self, deep: bool = True):
        params = {"estimator": self.estimator, "threshold": self.threshold}
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        if "threshold" in params:
            self.threshold = params.pop("threshold")
        est_params = {k[len("estimator__"):]: v for k, v in params.items() if k.startswith("estimator__")}
        other = {k: v for k, v in params.items() if not k.startswith("estimator__")}
        if est_params:
            self.estimator = clone(self.estimator)
            self.estimator.set_params(**est_params)
        if "estimator" in other:
            self.estimator = other["estimator"]
        return self

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


def normalize_metrics(metrics: Iterable[str] | None) -> List[str]:
    """Normalize metrics and filter to supported set."""
    if metrics is None:
        return ['auc', 'accuracy', 'aupr', 'f1']

    normalized = []
    seen = set()
    unsupported = []

    for metric in metrics:
        key = metric.lower()
        if key in SUPPORTED_METRICS:
            if key not in seen:
                normalized.append(key)
                seen.add(key)
        else:
            unsupported.append(metric)

    if unsupported:
        print(f"⚠️  Ignoring unsupported metrics: {unsupported}")

    if not normalized:
        normalized = ['auc']
        print("⚠️  No supported metrics provided. Defaulting to ['auc'].")

    return normalized


def get_imputer(method: str = 'mean', random_state: int = None, **kwargs):
    """
    Get imputer based on method name.

    Args:
        method: Imputation method ('mean', 'median', 'mice', 'svd', 'knn')
        random_state: Random seed
        **kwargs: Additional parameters for specific imputers

    Returns:
        Scikit-learn compatible imputer
    """
    if method == 'mean':
        return SimpleImputer(strategy='mean')
    elif method == 'median':
        return SimpleImputer(strategy='median')
    elif method == 'mice':
        estimator_type = kwargs.get('estimator', 'linear_regression')
        if estimator_type == 'linear_regression':
            estimator = LinearRegression()
        elif estimator_type == 'bayesian_ridge':
            estimator = BayesianRidge()
        else:
            estimator = None  # Use default

        return IterativeImputer(
            estimator=estimator,
            max_iter=kwargs.get('max_iter', 5),
            n_nearest_features=kwargs.get('n_nearest_features', None),
            skip_complete=kwargs.get('skip_complete', False),
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
    collect_predictions: bool = False,
    scoring: str = 'roc_auc',
    metrics: Iterable[str] | None = None,
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
        Dictionary with results including requested metrics (mean/std and per-fold lists)
        plus confusion-count summaries and best_params.
    """
    if imputation_params is None:
        imputation_params = {}

    metrics = normalize_metrics(metrics)

    # Initialize result storage
    metric_scores = {m: [] for m in metrics}
    tp_counts = []
    fp_counts = []
    fn_counts = []
    fold_sizes = []
    best_params_list = []
    feature_importances = []  # Track feature importance per fold
    coefficient_selection_indicators = []  # Track coefficient selection for logistic regression
    best_thresholds = []  # Track thresholds when optimizing F1

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
        model_clone = clone(model)
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = model_seed

        # If optimizing F1 and model supports predict_proba, wrap to tune threshold jointly
        tuned_param_grid = param_grid
        tuned_estimator = model_clone
        if scoring == 'f1' and hasattr(model_clone, 'predict_proba'):
            tuned_estimator = ThresholdWrapper(model_clone)
            tuned_param_grid = {f"estimator__{k}": v for k, v in param_grid.items()}
            tuned_param_grid['threshold'] = np.linspace(0.1, 0.9, 17)

        grid_search = GridSearchCV(
            estimator=tuned_estimator,
            param_grid=tuned_param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train_scaled, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        threshold_selected = best_params.get('threshold', 0.5)
        if scoring == 'f1' and hasattr(best_model, 'predict_proba'):
            best_thresholds.append(threshold_selected)

        if verbosity >= 1:
            # Format params nicely
            params_str = ", ".join([f"{k}={v}" for k, v in grid_search.best_params_.items()])
            print(f"    Selected params: {params_str}")
            if scoring == 'f1' and hasattr(best_model, 'predict_proba'):
                print(f"    Selected threshold for F1: {threshold_selected:.3f}")

        # Predict on test fold
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = best_model.decision_function(X_test_scaled)

        if scoring == 'f1' and hasattr(best_model, 'predict_proba'):
            y_pred = (y_pred_proba >= threshold_selected).astype(int)
        else:
            y_pred = best_model.predict(X_test_scaled)

        # Calculate metrics
        if 'auc' in metrics:
            auc = roc_auc_score(y_test, y_pred_proba)
            metric_scores['auc'].append(auc)
        if 'accuracy' in metrics:
            accuracy = accuracy_score(y_test, y_pred)
            metric_scores['accuracy'].append(accuracy)
        if 'aupr' in metrics:
            aupr = average_precision_score(y_test, y_pred_proba)
            metric_scores['aupr'].append(aupr)
        if 'f1' in metrics:
            f1 = f1_score(y_test, y_pred)
            metric_scores['f1'].append(f1)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        fold_size = len(y_test)

        tp_counts.append(tp)
        fp_counts.append(fp)
        fn_counts.append(fn)
        fold_sizes.append(fold_size)

        if verbosity >= 1:
            metric_str = []
            if 'auc' in metrics:
                metric_str.append(f"AUC: {metric_scores['auc'][-1]:.4f}")
            if 'accuracy' in metrics:
                metric_str.append(f"Accuracy: {metric_scores['accuracy'][-1]:.4f}")
            if 'aupr' in metrics:
                metric_str.append(f"AUPR: {metric_scores['aupr'][-1]:.4f}")
            if 'f1' in metrics:
                metric_str.append(f"F1: {metric_scores['f1'][-1]:.4f}")
            metrics_msg = ", ".join(metric_str)
            print(f"    Fold {metrics_msg}")

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
            # Also create selection indicator (1 if |coef| > 0.05, 0 otherwise)
            selection_indicator = (fold_importance > 0.05).astype(int)
            coefficient_selection_indicators.append(selection_indicator)

        if fold_importance is not None:
            feature_importances.append(fold_importance)

        if verbosity >= 2:
            debug_metrics = []
            for metric in metrics:
                value = metric_scores[metric][-1]
                debug_metrics.append(f"{metric.upper()}: {value:.4f}")
            print(f"    {', '.join(debug_metrics)}")

    # Aggregate results
    results = {
        'tp_counts': tp_counts,
        'fp_counts': fp_counts,
        'fn_counts': fn_counts,
        'fold_sizes': fold_sizes,
        'mean_tp': np.mean(tp_counts),
        'std_tp': np.std(tp_counts),
        'mean_fp': np.mean(fp_counts),
        'std_fp': np.std(fp_counts),
        'mean_fn': np.mean(fn_counts),
        'std_fn': np.std(fn_counts),
        'mean_tp_pct': np.mean([tp / fs * 100 for tp, fs in zip(tp_counts, fold_sizes)]) if fold_sizes else np.nan,
        'std_tp_pct': np.std([tp / fs * 100 for tp, fs in zip(tp_counts, fold_sizes)]) if fold_sizes else np.nan,
        'mean_fp_pct': np.mean([fp / fs * 100 for fp, fs in zip(fp_counts, fold_sizes)]) if fold_sizes else np.nan,
        'std_fp_pct': np.std([fp / fs * 100 for fp, fs in zip(fp_counts, fold_sizes)]) if fold_sizes else np.nan,
        'mean_fn_pct': np.mean([fn / fs * 100 for fn, fs in zip(fn_counts, fold_sizes)]) if fold_sizes else np.nan,
        'std_fn_pct': np.std([fn / fs * 100 for fn, fs in zip(fn_counts, fold_sizes)]) if fold_sizes else np.nan,
        'best_params': best_params_list
    }

    for metric, scores in metric_scores.items():
        results[f'{metric}_scores'] = scores
        results[f'mean_{metric}'] = np.mean(scores)
        results[f'std_{metric}'] = np.std(scores)

    if best_thresholds:
        results['best_thresholds'] = best_thresholds
        results['mean_threshold'] = float(np.mean(best_thresholds))
        results['std_threshold'] = float(np.std(best_thresholds))

    # Add aggregated feature importance if available
    if feature_importances:
        # Average importance across folds
        mean_importance = np.mean(feature_importances, axis=0)
        results['feature_importance'] = mean_importance
        # Also save raw fold-level importances for detailed analysis
        results['feature_importances_raw'] = feature_importances

    # Add selection frequency if available (for logistic regression)
    if coefficient_selection_indicators:
        # Calculate mean selection frequency across folds
        mean_selection_frequency = np.mean(coefficient_selection_indicators, axis=0)
        results['selection_frequency'] = mean_selection_frequency
        # Also save raw fold-level indicators for detailed analysis
        results['coefficient_selection_indicators_raw'] = coefficient_selection_indicators

    # Add fold predictions if requested
    if collect_predictions:
        results['fold_predictions'] = fold_predictions

    if verbosity >= 1:
        for metric in metrics:
            mean_key = f"mean_{metric}"
            std_key = f"std_{metric}"
            if mean_key in results and std_key in results:
                print(f"  Mean {metric.upper()}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")

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
    collect_predictions: bool = False,
    scoring: str = 'roc_auc',
    metrics: Iterable[str] | None = None,
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
        metrics: Metrics to compute (subset of {'auc','accuracy','aupr','f1'})

    Returns:
        Dictionary with aggregated results across all iterations
    """
    metrics = normalize_metrics(metrics)
    # Generate seeds for each iteration
    rng = np.random.RandomState(master_seed)
    seeds = rng.randint(0, 10000, size=n_iterations)

    # Storage for results
    all_metric_scores = {m: [] for m in metrics}
    all_tp_counts = []
    all_fp_counts = []
    all_fn_counts = []
    all_fold_sizes = []
    all_feature_importances = []
    all_selection_frequencies = []  # Track selection frequencies for logistic regression
    all_feature_importances_raw = []  # Raw fold-level importances from all iterations
    all_selection_indicators_raw = []  # Raw fold-level indicators from all iterations
    all_thresholds = []

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
            collect_predictions=collect_predictions,
            scoring=scoring,
            metrics=metrics
        )

        for metric in metrics:
            mean_key = f'mean_{metric}'
            if mean_key in results:
                all_metric_scores[metric].append(results[mean_key])
        all_tp_counts.extend(results.get('tp_counts', []))
        all_fp_counts.extend(results.get('fp_counts', []))
        all_fn_counts.extend(results.get('fn_counts', []))
        all_fold_sizes.extend(results.get('fold_sizes', []))

        # Collect feature importance if available
        if 'feature_importance' in results:
            all_feature_importances.append(results['feature_importance'])

        # Collect raw fold-level importances if available
        if 'feature_importances_raw' in results:
            # Extend the list with all folds from this iteration
            all_feature_importances_raw.extend(results['feature_importances_raw'])

        # Collect selection frequency if available (for logistic regression)
        if 'selection_frequency' in results:
            all_selection_frequencies.append(results['selection_frequency'])

        # Collect raw fold-level indicators if available
        if 'coefficient_selection_indicators_raw' in results:
            # Extend the list with all folds from this iteration
            all_selection_indicators_raw.extend(results['coefficient_selection_indicators_raw'])

        # Collect fold predictions if requested
        if collect_predictions and 'fold_predictions' in results:
            iteration_predictions[f'iteration_{i}'] = results['fold_predictions']

        if 'best_thresholds' in results:
            all_thresholds.extend(results['best_thresholds'])

    # Aggregate results
    aggregated_results = {
        'tp_counts': all_tp_counts,
        'fp_counts': all_fp_counts,
        'fn_counts': all_fn_counts,
        'fold_sizes': all_fold_sizes,
        'mean_tp': np.mean(all_tp_counts) if all_tp_counts else np.nan,
        'std_tp': np.std(all_tp_counts) if all_tp_counts else np.nan,
        'mean_fp': np.mean(all_fp_counts) if all_fp_counts else np.nan,
        'std_fp': np.std(all_fp_counts) if all_fp_counts else np.nan,
        'mean_fn': np.mean(all_fn_counts) if all_fn_counts else np.nan,
        'std_fn': np.std(all_fn_counts) if all_fn_counts else np.nan,
        'mean_tp_pct': np.mean([tp / fs * 100 for tp, fs in zip(all_tp_counts, all_fold_sizes)]) if all_fold_sizes else np.nan,
        'std_tp_pct': np.std([tp / fs * 100 for tp, fs in zip(all_tp_counts, all_fold_sizes)]) if all_fold_sizes else np.nan,
        'mean_fp_pct': np.mean([fp / fs * 100 for fp, fs in zip(all_fp_counts, all_fold_sizes)]) if all_fold_sizes else np.nan,
        'std_fp_pct': np.std([fp / fs * 100 for fp, fs in zip(all_fp_counts, all_fold_sizes)]) if all_fold_sizes else np.nan,
        'mean_fn_pct': np.mean([fn / fs * 100 for fn, fs in zip(all_fn_counts, all_fold_sizes)]) if all_fold_sizes else np.nan,
        'std_fn_pct': np.std([fn / fs * 100 for fn, fs in zip(all_fn_counts, all_fold_sizes)]) if all_fold_sizes else np.nan,
        'seeds': seeds.tolist()
    }

    for metric, scores in all_metric_scores.items():
        aggregated_results[f'{metric}_scores'] = scores
        aggregated_results[f'mean_{metric}'] = np.mean(scores)
        aggregated_results[f'std_{metric}'] = np.std(scores)

    if all_thresholds:
        aggregated_results['best_thresholds'] = all_thresholds
        aggregated_results['mean_threshold'] = float(np.mean(all_thresholds))
        aggregated_results['std_threshold'] = float(np.std(all_thresholds))

    # Add feature importance if available and feature names provided
    if all_feature_importances and feature_names is not None:
        # Average importance across all iterations
        mean_importance = np.mean(all_feature_importances, axis=0)
        std_importance = np.std(all_feature_importances, axis=0)
        aggregated_results['feature_importance'] = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_importance,
            'importance_std': std_importance
        }).sort_values('importance', ascending=False)

        # Save raw iteration-level importance data (averaged across folds within iteration)
        importance_iterations_df = pd.DataFrame(
            all_feature_importances,
            columns=feature_names
        )
        importance_iterations_df.insert(0, 'iteration', range(len(all_feature_importances)))
        aggregated_results['feature_importance_iterations'] = importance_iterations_df

        # Save raw fold-level importance data (all folds from all iterations)
        if all_feature_importances_raw:
            importance_folds_df = pd.DataFrame(
                all_feature_importances_raw,
                columns=feature_names
            )
            importance_folds_df.insert(0, 'fold', range(len(all_feature_importances_raw)))
            aggregated_results['feature_importance_folds'] = importance_folds_df

    # Add selection frequency if available and feature names provided
    if all_selection_frequencies and feature_names is not None:
        # Average selection frequency across all iterations
        mean_selection_frequency = np.mean(all_selection_frequencies, axis=0)
        std_selection_frequency = np.std(all_selection_frequencies, axis=0)
        aggregated_results['selection_frequency'] = pd.DataFrame({
            'feature': feature_names,
            'selection_frequency': mean_selection_frequency,
            'selection_frequency_std': std_selection_frequency
        }).sort_values('selection_frequency', ascending=False)

        # Save raw iteration-level selection frequency data (averaged across folds within iteration)
        selection_freq_iterations_df = pd.DataFrame(
            all_selection_frequencies,
            columns=feature_names
        )
        selection_freq_iterations_df.insert(0, 'iteration', range(len(all_selection_frequencies)))
        aggregated_results['selection_frequency_iterations'] = selection_freq_iterations_df

        # Save raw fold-level selection indicators (all folds from all iterations)
        if all_selection_indicators_raw:
            selection_indicators_folds_df = pd.DataFrame(
                all_selection_indicators_raw,
                columns=feature_names
            )
            selection_indicators_folds_df.insert(0, 'fold', range(len(all_selection_indicators_raw)))
            aggregated_results['selection_frequency_folds'] = selection_indicators_folds_df

    # Add iteration predictions if requested
    if collect_predictions:
        aggregated_results['iteration_predictions'] = iteration_predictions

    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"AGGREGATED RESULTS ({n_iterations} iterations)")
        print(f"{'='*80}")
        for metric in metrics:
            mean_key = f'mean_{metric}'
            std_key = f'std_{metric}'
            if mean_key in aggregated_results and std_key in aggregated_results:
                print(f"{metric.upper()}: {aggregated_results[mean_key]:.4f} ± {aggregated_results[std_key]:.4f}")

    return aggregated_results
