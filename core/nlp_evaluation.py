"""
NLP Feature Evaluation with Embedding and Dimensionality Selection.

This module implements nested cross-validation for NLP features (BoW, TF-IDF, embeddings)
with automatic selection of best embedding type and dimensionality during inner CV.

Based on evaluation_code_0701.ipynb nested_cv_evaluation_nlp function.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple, Iterable
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

# Import custom logging print function
from core.log_utils import print
from core.evaluation import normalize_metrics, get_imputer


class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to treat threshold as a hyperparameter."""
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


def create_embedding_variants(
    embedding_df: pd.DataFrame,
    embedding_name: str,
    target_dims: List[int] = [50, 100, 200]
) -> Dict[str, pd.DataFrame]:
    """
    Create multiple dimensionality variants of an embedding.

    If target_dims includes 'ALL' (or -1), include the original embedding.
    Otherwise, only create trimmed versions.

    For each target dimension:
    - If embedding has more dimensions than target, create trimmed version (first N columns)
    - Otherwise, keep original

    Args:
        embedding_df: Original embedding DataFrame
        embedding_name: Name of the embedding
        target_dims: List of target dimensionalities to create (can include 'ALL' or -1 for original)

    Returns:
        Dictionary mapping variant names to DataFrames
    """
    variants = {}
    current_dims = embedding_df.shape[1]

    # Check if 'ALL' or -1 is in target_dims
    include_original = 'ALL' in target_dims or -1 in target_dims

    if include_original:
        variants[embedding_name] = embedding_df  # Original

    for target_dim in target_dims:
        # Skip 'ALL' and -1 markers
        if target_dim == 'ALL' or target_dim == -1:
            continue

        if current_dims > target_dim:
            # Trimmed version - first target_dim columns
            trimmed = embedding_df.iloc[:, :target_dim].copy()
            trimmed.columns = [f'{embedding_name}_dim{i}' for i in range(target_dim)]
            variants[f'{embedding_name}_trimmed{target_dim}'] = trimmed

    return variants


def inner_cv_nlp(
    embedding_variants: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
    y: pd.Series,
    model,
    param_grid: Dict[str, Any],
    inner_cv,
    scaler: StandardScaler,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_seed: int = None,
    scoring: str = 'roc_auc'
) -> Tuple[str, float, float, float, float, int, int, int, int, Any, np.ndarray, np.ndarray, Dict[str, Any], float]:
    """
    Inner CV loop that selects best embedding variant and hyperparameters.

    For each embedding variant:
    1. Combine with baseline features
    2. Split into train/test using provided indices
    3. Scale features
    4. Run GridSearchCV for hyperparameter tuning
    5. Track best performing variant

    Note: No imputation is done here. Baseline features should be pre-imputed
    before calling this function. Embeddings should not have missing values.

    Args:
        embedding_variants: Dict mapping variant names to embedding DataFrames
        baseline_df: Baseline features DataFrame (should be pre-imputed)
        y: Target labels
        model: Model to train
        param_grid: Hyperparameter grid
        inner_cv: Cross-validator for inner CV
        scaler: Scaler instance
        train_idx: Training indices
        test_idx: Test indices

    Returns:
        Tuple of (best_variant_name, test_auc, test_accuracy, test_aupr, test_f1,
                  test_tp, test_fp, test_fn, fold_size, feature_importance,
                  y_true, y_pred_proba, best_params, best_threshold)
    """
    best_overall_score = -1
    best_model = None
    best_variant_name = None
    best_X_test_processed = None
    best_cv_result = None
    best_params = None
    best_threshold = 0.5

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Try each embedding variant
    for variant_name, embedding_df in embedding_variants.items():
        # Combine baseline + embedding
        X_combined = pd.concat([baseline_df, embedding_df], axis=1)
        X_combined.columns = X_combined.columns.astype(str)

        # Split train/test
        X_train = X_combined.iloc[train_idx, :]
        X_test = X_combined.iloc[test_idx, :]

        # Create new scaler instance to avoid data leakage
        current_scaler = clone(scaler)

        # Fit scaler on training data
        X_train_processed = pd.DataFrame(
            current_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        # Transform test data
        X_test_processed = pd.DataFrame(
            current_scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        # Grid search for hyperparameter tuning
        # Set model random_state if the model has this parameter
        model_clone = clone(model)
        if hasattr(model_clone, 'random_state') and random_seed is not None:
            model_clone.random_state = random_seed

        tuned_estimator = model_clone
        tuned_param_grid = param_grid
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
            verbose=0,
            return_train_score=True
        )

        # Fit on training data
        grid_search.fit(X_train_processed, y_train)

        # Check if this variant is better than current best
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_variant_name = variant_name
            best_X_test_processed = X_test_processed
            best_cv_result = grid_search.cv_results_
            best_params = grid_search.best_params_
            threshold_selected = best_params.get('threshold', 0.5)
            best_threshold = threshold_selected

    # Evaluate best model on test set
    y_pred_proba = best_model.predict_proba(best_X_test_processed)[:, 1]
    if scoring == 'f1' and hasattr(best_model, 'predict_proba'):
        y_pred = (y_pred_proba >= best_threshold).astype(int)
    else:
        y_pred = best_model.predict(best_X_test_processed)

    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Get feature importance if available
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': best_X_test_processed.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(best_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': best_X_test_processed.columns,
            'importance': np.abs(best_model.coef_[0])
        }).sort_values('importance', ascending=False)

    test_aupr = average_precision_score(y_test, y_pred_proba)
    test_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # Return predictions along with other results
    return (
        best_variant_name,
        test_auc,
        test_accuracy,
        test_aupr,
        test_f1,
        tp,
        fp,
        fn,
        len(y_test),
        feature_importance,
        y_test.values,
        y_pred_proba,
        best_params,
        best_threshold,
    )


def inner_cv_nlp_model_selection(
    embedding_variants: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
    y: pd.Series,
    models_config: Dict[str, Dict],
    inner_cv,
    scaler: StandardScaler,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_seed: int = None,
    scoring: str = 'roc_auc'
) -> Tuple[str, str, float, float, float, float, int, int, int, int, Any, np.ndarray, np.ndarray, Dict[str, Any], float]:
    """
    Inner CV loop that selects best MODEL TYPE + embedding variant + hyperparameters.

    For each model type:
        For each embedding variant:
            1. Combine with baseline features
            2. Split into train/test using provided indices
            3. Scale features
            4. Run GridSearchCV for hyperparameter tuning
            5. Track best performing combination

    Args:
        embedding_variants: Dict mapping variant names to embedding DataFrames
        baseline_df: Baseline features DataFrame (should be pre-imputed)
        y: Target labels
        models_config: Dictionary of model configurations (multiple models)
        inner_cv: Cross-validator for inner CV
        scaler: Scaler instance
        train_idx: Training indices
        test_idx: Test indices
        random_seed: Random seed

    Returns:
        Tuple of (best_model_name, best_variant_name, test_auc, test_accuracy,
                  test_aupr, test_f1, test_tp, test_fp, test_fn, fold_size,
                  feature_importance, y_true, y_pred_proba, best_params, best_threshold)
    """
    import importlib

    def import_model_class(class_path: str):
        """Import model class from string path."""
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    best_overall_score = -1
    best_model_instance = None
    best_model_name = None
    best_variant_name = None
    best_X_test_processed = None
    best_params = None
    best_threshold = 0.5
    all_scores = {}

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Try each model type
    for model_name, model_config in models_config.items():
        model_scores = {}

        # Import model class
        model_class = import_model_class(model_config['class'])

        # Get base parameters
        params = model_config['params'].copy()
        if 'random_state' in params and params['random_state'] is None:
            params['random_state'] = random_seed

        # Try each embedding variant
        for variant_name, embedding_df in embedding_variants.items():
            # Combine baseline + embedding
            X_combined = pd.concat([baseline_df, embedding_df], axis=1)
            X_combined.columns = X_combined.columns.astype(str)

            # Split train/test
            X_train = X_combined.iloc[train_idx, :]
            X_test = X_combined.iloc[test_idx, :]

            # Create new scaler instance
            current_scaler = clone(scaler)

            # Fit scaler on training data
            X_train_processed = pd.DataFrame(
                current_scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )

            # Transform test data
            X_test_processed = pd.DataFrame(
                current_scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            # Create model instance
            model = model_class(**params)

            # Grid search for hyperparameter tuning (with optional threshold)
            param_grid = model_config['param_grid']
            tuned_estimator = model
            tuned_param_grid = param_grid
            if scoring == 'f1' and hasattr(model, 'predict_proba'):
                tuned_estimator = ThresholdWrapper(model)
                tuned_param_grid = {f"estimator__{k}": v for k, v in param_grid.items()}
                tuned_param_grid['threshold'] = np.linspace(0.1, 0.9, 17)

            grid_search = GridSearchCV(
                tuned_estimator,
                tuned_param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=-1,
                error_score='raise'
            )

            try:
                grid_search.fit(X_train_processed, y_train)
                score = grid_search.best_score_
                model_scores[variant_name] = score

                threshold_selected = 0.5
                if scoring == 'f1' and hasattr(grid_search.best_estimator_, 'predict_proba'):
                    thresh_cv = inner_cv
                    n_repeats = getattr(inner_cv, 'n_repeats', 1)
                    if n_repeats and n_repeats > 1:
                        thresh_cv = StratifiedKFold(
                            n_splits=inner_cv.n_splits,
                            shuffle=True,
                            random_state=getattr(inner_cv, 'random_state', None)
                        )

                    thresholds = np.linspace(0.1, 0.9, 17)
                    cv_probs = cross_val_predict(
                        grid_search.best_estimator_,
                        X_train_processed,
                        y_train,
                        cv=thresh_cv,
                        method='predict_proba'
                    )[:, 1]
                    best_f1_val = -1
                    for t in thresholds:
                        preds_t = (cv_probs >= t).astype(int)
                        f1_val = f1_score(y_train, preds_t)
                        if f1_val > best_f1_val:
                            best_f1_val = f1_val
                            threshold_selected = t

                # Update best if this combination is better
                if score > best_overall_score:
                    best_overall_score = score
                    best_model_instance = grid_search.best_estimator_
                    best_model_name = model_name
                    best_variant_name = variant_name
                    best_X_test_processed = X_test_processed
                    best_params = grid_search.best_params_
                    best_threshold = threshold_selected

            except Exception as e:
                model_scores[variant_name] = np.nan
                continue

        all_scores[model_name] = model_scores

    if best_model_instance is None:
        raise RuntimeError("All model/variant combinations failed during inner CV")

    # Evaluate best model on test set
    y_pred_proba = best_model_instance.predict_proba(best_X_test_processed)[:, 1]
    if scoring == 'f1':
        y_pred = (y_pred_proba >= best_threshold).astype(int)
    else:
        y_pred = best_model_instance.predict(best_X_test_processed)

    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_aupr = average_precision_score(y_test, y_pred_proba)
    test_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # Extract feature importance
    feature_importance = None
    if hasattr(best_model_instance, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': best_X_test_processed.columns,
            'importance': best_model_instance.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(best_model_instance, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': best_X_test_processed.columns,
            'importance': np.abs(best_model_instance.coef_[0])
        }).sort_values('importance', ascending=False)

    return (best_model_name, best_variant_name, test_auc, test_accuracy,
            test_aupr, test_f1, tp, fp, fn, len(y_test), feature_importance, y_test.values, y_pred_proba, best_params, best_threshold)


def run_multiple_evaluations_nlp(
    embedding_df: pd.DataFrame,
    embedding_name: str,
    baseline_df: pd.DataFrame,
    y: pd.Series,
    model,
    param_grid: Dict[str, Any],
    n_iterations: int = 100,
    master_seed: int = 42,
    n_outer: int = 3,
    n_inner: int = 3,
    n_inner_repeats: int = 1,
    target_dims: List[int] = [50, 100, 200],
    imputation_method: str = 'mean',
    imputation_params: Dict[str, Any] = None,
    verbosity: int = 1,
    collect_predictions: bool = False,  # Not currently used (NLP features excluded from permutation tests)
    scoring: str = 'roc_auc',
    metrics: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Run nested CV evaluation multiple times with dimensionality selection for NLP embeddings.

    Same structure as run_multiple_evaluations(), but with dimension selection in inner CV.

    Args:
        embedding_df: Embedding features DataFrame
        embedding_name: Name of the embedding
        baseline_df: Baseline features DataFrame
        y: Target labels
        model: Model to evaluate
        param_grid: Hyperparameter grid
        n_iterations: Number of evaluation iterations
        master_seed: Master seed for generating iteration seeds
        n_outer: Number of outer CV folds
        n_inner: Number of inner CV folds
        n_inner_repeats: Number of inner CV repeats
        target_dims: List of target dimensionalities for variants
        imputation_method: Imputation method ('mean', 'median', 'mice', 'svd', 'knn')
        imputation_params: Imputer parameters
        verbosity: Verbosity level
        collect_predictions: Whether to collect predictions (not used; NLP excluded from permutation tests)

    Returns:
        Dictionary with aggregated results across all iterations
    """
    metrics = normalize_metrics(metrics)
    # Create embedding variants (done once, used in all iterations)
    embedding_variants = create_embedding_variants(
        embedding_df=embedding_df,
        embedding_name=embedding_name,
        target_dims=target_dims
    )

    if verbosity >= 1:
        print(f"  Created {len(embedding_variants)} embedding variants:")
        for variant_name, variant_df in embedding_variants.items():
            print(f"    - {variant_name}: {variant_df.shape[1]} dimensions")

    # Generate seeds for each iteration
    rng = np.random.RandomState(master_seed)
    seeds = rng.randint(0, 10000, size=n_iterations)

    # Storage for results
    all_metric_scores = {m: [] for m in metrics}
    all_tp_counts = []
    all_fp_counts = []
    all_fn_counts = []
    all_fold_sizes = []
    all_best_variants = []
    all_thresholds = []

    # Initialize iteration predictions storage if requested
    if collect_predictions:
        iteration_predictions = {}

    # Run evaluations
    for i, seed in enumerate(seeds):
        if verbosity >= 1:
            print(f"\nIteration {i + 1}/{n_iterations} (seed={seed})")

        # Create outer CV with iteration-specific seed
        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

        # Impute baseline features once per iteration (embeddings have no NaN)
        # Use iteration seed for imputation (happens once per iteration, not per fold)
        imputer_params = imputation_params if imputation_params else {}
        imputer = get_imputer(imputation_method, random_state=seed, **imputer_params)
        baseline_df_imputed = pd.DataFrame(
            imputer.fit_transform(baseline_df),
            columns=baseline_df.columns,
            index=baseline_df.index
        )

        # Storage for this iteration's folds
        iteration_metric_scores = {m: [] for m in metrics}
        iteration_tp_counts = []
        iteration_fp_counts = []
        iteration_fn_counts = []
        iteration_fold_sizes = []
        iteration_best_variants = []
        iteration_thresholds = []

        # Initialize fold predictions storage for this iteration if requested
        if collect_predictions:
            fold_predictions = {}

        # Create RNG for generating fold-specific seeds
        fold_rng = np.random.RandomState(seed)

        # Outer CV loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(baseline_df_imputed, y)):
            if verbosity >= 2:
                print(f"  Outer fold {fold_idx + 1}/{n_outer}")

            # Generate fold-specific seed and derive component-specific seeds
            fold_seed = fold_rng.randint(0, 100000)
            component_rng = np.random.RandomState(fold_seed)
            inner_cv_seed = component_rng.randint(0, 100000)
            model_seed = component_rng.randint(0, 100000)

            # Create inner CV with fold-specific seed
            if n_inner_repeats > 1:
                inner_cv = RepeatedStratifiedKFold(
                    n_splits=n_inner,
                    n_repeats=n_inner_repeats,
                    random_state=inner_cv_seed
                )
            else:
                inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=inner_cv_seed)

            # Create scaler for this fold
            scaler = StandardScaler()

            # Inner CV: select best variant and hyperparameters
            (best_variant, test_auc, test_accuracy, test_aupr, test_f1,
             test_tp, test_fp, test_fn, fold_size, feature_importance, y_true, y_pred_proba, best_params, threshold_selected) = inner_cv_nlp(
                embedding_variants=embedding_variants,
                baseline_df=baseline_df_imputed,
                y=y,
                model=model,
                param_grid=param_grid,
                inner_cv=inner_cv,
                scaler=scaler,
                train_idx=train_idx,
                test_idx=test_idx,
                random_seed=model_seed,
                scoring=scoring
            )

            if verbosity >= 1:
                # Format params nicely
                params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
                print(f"  Outer fold {fold_idx + 1}/{n_outer}")
                print(f"    Selected variant: {best_variant}")
                print(f"    Selected params: {params_str}")
                fold_metric_parts = []
                fold_values = {
                    'auc': test_auc,
                    'accuracy': test_accuracy,
                    'aupr': test_aupr,
                    'f1': test_f1
                }
                for metric in metrics:
                    fold_metric_parts.append(f"{metric.upper()}: {fold_values[metric]:.4f}")
                print(f"    Fold {', '.join(fold_metric_parts)}")

            # Store fold results
            metric_values = {
                'auc': test_auc,
                'accuracy': test_accuracy,
                'aupr': test_aupr,
                'f1': test_f1
            }
            for metric in metrics:
                iteration_metric_scores[metric].append(metric_values[metric])
            iteration_tp_counts.append(test_tp)
            iteration_fp_counts.append(test_fp)
            iteration_fn_counts.append(test_fn)
            iteration_fold_sizes.append(fold_size)
            iteration_best_variants.append(best_variant)
            if scoring == 'f1':
                iteration_thresholds.append(threshold_selected)

            # Store predictions if requested
            if collect_predictions:
                fold_predictions[fold_idx] = {
                    'y_true': y_true.copy(),
                    'y_pred': y_pred_proba.copy(),
                    'test_idx': test_idx.copy()
                }

        # Store iteration results (mean across folds for this iteration)
        for metric in metrics:
            all_metric_scores[metric].append(np.mean(iteration_metric_scores[metric]))
        all_tp_counts.extend(iteration_tp_counts)
        all_fp_counts.extend(iteration_fp_counts)
        all_fn_counts.extend(iteration_fn_counts)
        all_fold_sizes.extend(iteration_fold_sizes)
        all_best_variants.extend(iteration_best_variants)
        if scoring == 'f1':
            all_thresholds.extend(iteration_thresholds)

        # Store fold predictions for this iteration
        if collect_predictions:
            iteration_predictions[f'iteration_{i}'] = fold_predictions

        # Log selected variants for this iteration
        if verbosity >= 1:
            from collections import Counter
            variant_counts = Counter(iteration_best_variants)
            variants_summary = ", ".join([f"{v}({c})" for v, c in variant_counts.items()])
            print(f"  Selected variants: {variants_summary}")
            metric_summary = []
            for metric in metrics:
                metric_summary.append(f"{metric.upper()}: {np.mean(iteration_metric_scores[metric]):.4f}")
            print(f"  Mean {'; '.join(metric_summary)}")

    # Calculate summary statistics (same format as standard evaluation)
    results = {
        'tp_counts': all_tp_counts,
        'fp_counts': all_fp_counts,
        'fn_counts': all_fn_counts,
        'fold_sizes': all_fold_sizes,
        'best_variants': all_best_variants,
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
        'seeds': seeds.tolist()  # Add seeds for reproducibility
    }

    if scoring == 'f1' and all_thresholds:
        results['thresholds'] = all_thresholds
        results['mean_threshold'] = float(np.mean(all_thresholds))
        results['std_threshold'] = float(np.std(all_thresholds))
        results['mean_threshold'] = float(np.mean(all_thresholds))
        results['std_threshold'] = float(np.std(all_thresholds))

    for metric, scores in all_metric_scores.items():
        results[f'{metric}_scores'] = scores
        results[f'mean_{metric}'] = np.mean(scores)
        results[f'std_{metric}'] = np.std(scores)

    # Add iteration predictions if requested
    if collect_predictions:
        results['iteration_predictions'] = iteration_predictions

    # Print final summary
    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        for metric in metrics:
            mean_key = f"mean_{metric}"
            std_key = f"std_{metric}"
            if mean_key in results and std_key in results:
                print(f"  Mean {metric.upper()}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")

        # Show most common variants
        from collections import Counter
        variant_counts = Counter(all_best_variants)
        print(f"\n  Most frequently selected variants:")
        for variant, count in variant_counts.most_common(5):
            percentage = (count / len(all_best_variants)) * 100
            print(f"    {variant}: {count}/{len(all_best_variants)} ({percentage:.1f}%)")
        print(f"{'='*80}")

    return results


def run_multiple_evaluations_nlp_model_selection(
    embedding_df: pd.DataFrame,
    embedding_name: str,
    baseline_df: pd.DataFrame,
    y: pd.Series,
    models_config: Dict[str, Dict],
    n_iterations: int = 100,
    master_seed: int = None,
    n_outer: int = 3,
    n_inner: int = 3,
    n_inner_repeats: int = 1,
    target_dims: List[int] = [50, 100, 200],
    imputation_method: str = 'mean',
    imputation_params: Dict[str, Any] = None,
    verbosity: int = 0,
    collect_predictions: bool = False,
    scoring: str = 'roc_auc',
    metrics: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Run nested CV with MODEL SELECTION + dimensionality selection for NLP features.

    This extends `run_multiple_evaluations_nlp` to also select the best MODEL TYPE
    in addition to the best embedding dimensionality.

    Args:
        embedding_df: NLP embedding DataFrame (all dimensions)
        embedding_name: Name of the embedding
        baseline_df: Baseline features DataFrame
        y: Target labels (pd.Series)
        models_config: Dictionary of model configurations (multiple models)
        n_iterations: Number of outer CV iterations
        master_seed: Master random seed
        n_outer: Number of outer CV folds
        n_inner: Number of inner CV folds
        n_inner_repeats: Number of inner CV repeats
        target_dims: List of target dimensionalities to try
        imputation_method: Imputation method ('mean', 'median', 'mice', 'svd', 'knn')
        imputation_params: Additional imputation parameters
        verbosity: Verbosity level
        collect_predictions: Whether to collect predictions

    Returns:
        Dictionary with results including model selection statistics
    """
    from collections import Counter

    metrics = normalize_metrics(metrics)

    if imputation_params is None:
        imputation_params = {}

    # Create embedding variants with different dimensionalities
    embedding_variants = create_embedding_variants(embedding_df, embedding_name, target_dims)

    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"NLP EVALUATION WITH MODEL SELECTION")
        print(f"{'='*80}")
        print(f"Embedding: {embedding_name}")
        print(f"Candidate models: {list(models_config.keys())}")
        print(f"Variants created: {list(embedding_variants.keys())}")
        print(f"{'='*80}\n")

    # Storage for results
    all_auc_scores = []
    all_accuracy_scores = []
    all_aupr_scores = []
    all_f1_scores = []
    all_tp_counts = []
    all_fp_counts = []
    all_fn_counts = []
    all_fold_sizes = []
    all_best_models = []
    all_best_variants = []
    all_best_params = []
    all_thresholds = []

    if collect_predictions:
        iteration_predictions = {}

    seeds = [None if master_seed is None else master_seed + i * 1000 for i in range(n_iterations)]

    # Outer CV iterations
    for i, iteration_seed in enumerate(seeds):

        if verbosity >= 1:
            print(f"Iteration {i+1}/{n_iterations} (seed={iteration_seed})")

        # Outer CV for performance estimation
        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=iteration_seed)

        # Inner CV for model + variant selection
        if n_inner_repeats > 1:
            inner_cv = RepeatedStratifiedKFold(
                n_splits=n_inner,
                n_repeats=n_inner_repeats,
                random_state=iteration_seed
            )
        else:
            inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=iteration_seed)

        if collect_predictions:
            fold_predictions = {}

        # Impute baseline features
        imputer_params = imputation_params if imputation_params else {}
        imputer = get_imputer(imputation_method, random_state=iteration_seed, **imputer_params)

        baseline_imputed = pd.DataFrame(
            imputer.fit_transform(baseline_df),
            index=baseline_df.index,
            columns=baseline_df.columns
        )

        scaler = StandardScaler()

        # Outer CV loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(baseline_df, y)):
            fold_seed = None if iteration_seed is None else iteration_seed + fold_idx

            # Inner CV: Model + variant selection
            (best_model_name, best_variant_name, test_auc, test_accuracy,
             test_aupr, test_f1, test_tp, test_fp, test_fn, fold_size,
             feature_importance, y_true, y_pred_proba, best_params, threshold_selected) = inner_cv_nlp_model_selection(
                embedding_variants=embedding_variants,
                baseline_df=baseline_imputed,
                y=y,
                models_config=models_config,
                inner_cv=inner_cv,
                scaler=scaler,
                train_idx=train_idx,
                test_idx=test_idx,
                random_seed=fold_seed,
                scoring=scoring
            )

            all_auc_scores.append(test_auc)
            all_accuracy_scores.append(test_accuracy)
            all_aupr_scores.append(test_aupr)
            all_f1_scores.append(test_f1)
            all_tp_counts.append(test_tp)
            all_fp_counts.append(test_fp)
            all_fn_counts.append(test_fn)
            all_fold_sizes.append(fold_size)
            all_best_models.append(best_model_name)
            all_best_variants.append(best_variant_name)
            all_best_params.append(best_params)
            if scoring == 'f1':
                all_thresholds.append(threshold_selected)

            if verbosity >= 2:
                print(f"  Fold {fold_idx+1}: {best_model_name} + {best_variant_name} -> AUC={test_auc:.4f}")

            # Store predictions if requested
            if collect_predictions:
                fold_predictions[fold_idx] = {
                    'y_true': y_true.copy(),
                    'y_pred': y_pred_proba.copy(),
                    'test_idx': test_idx.copy()
                }

        if collect_predictions:
            iteration_predictions[f'iteration_{i}'] = fold_predictions

        if verbosity >= 1:
            auc_slice = all_auc_scores[i*n_outer:(i+1)*n_outer]
            acc_slice = all_accuracy_scores[i*n_outer:(i+1)*n_outer]
            aupr_slice = all_aupr_scores[i*n_outer:(i+1)*n_outer]
            f1_slice = all_f1_scores[i*n_outer:(i+1)*n_outer]
            metric_slices = {
                'auc': auc_slice,
                'accuracy': acc_slice,
                'aupr': aupr_slice,
                'f1': f1_slice
            }
            for metric in metrics:
                print(f"  Iteration {i+1} mean {metric.upper()}: {np.mean(metric_slices[metric]):.4f}")
            print()

    # Aggregate results
    model_counts = Counter(all_best_models)
    variant_counts = Counter(all_best_variants)

    results = {
        'all_auc_scores': all_auc_scores,
        'all_accuracy_scores': all_accuracy_scores,
        'all_aupr_scores': all_aupr_scores,
        'all_f1_scores': all_f1_scores,
        'tp_counts': all_tp_counts,
        'fp_counts': all_fp_counts,
        'fn_counts': all_fn_counts,
        'fold_sizes': all_fold_sizes,
        'mean_auc': np.mean(all_auc_scores),
        'std_auc': np.std(all_auc_scores),
        'mean_accuracy': np.mean(all_accuracy_scores),
        'std_accuracy': np.std(all_accuracy_scores),
        'mean_aupr': np.mean(all_aupr_scores),
        'std_aupr': np.std(all_aupr_scores),
        'mean_f1': np.mean(all_f1_scores),
        'std_f1': np.std(all_f1_scores),
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
        'selected_models': all_best_models,
        'selected_variants': all_best_variants,
        'selected_params': all_best_params,
        'model_selection_counts': model_counts,
        'variant_selection_counts': variant_counts,
        'seeds': seeds.tolist()  # Add seeds for reproducibility
    }

    if collect_predictions:
        results['iteration_predictions'] = iteration_predictions

    if scoring == 'f1' and all_thresholds:
        results['thresholds'] = all_thresholds
        results['mean_threshold'] = float(np.mean(all_thresholds))
        results['std_threshold'] = float(np.std(all_thresholds))

    # Print summary
    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"MODEL SELECTION SUMMARY - {embedding_name}")
        print(f"{'='*80}")
        for metric in metrics:
            mean_key = f"mean_{metric}"
            std_key = f"std_{metric}"
            if mean_key in results and std_key in results:
                print(f"Mean {metric.upper()}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")
        print(f"\nModel Selection (top 5):")
        for model, count in model_counts.most_common(5):
            percentage = (count / len(all_best_models)) * 100
            print(f"  {model}: {count}/{len(all_best_models)} ({percentage:.1f}%)")
        print(f"\nDimensionality Selection (top 5):")
        for variant, count in variant_counts.most_common(5):
            percentage = (count / len(all_best_variants)) * 100
            print(f"  {variant}: {count}/{len(all_best_variants)} ({percentage:.1f}%)")
        print(f"{'='*80}\n")

    return results


def run_multiple_evaluations_rfg(
    embeddings_dict: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
    y: pd.Series,
    model,
    param_grid: Dict[str, Any],
    n_iterations: int = 100,
    master_seed: int = 42,
    n_outer: int = 3,
    n_inner: int = 3,
    n_inner_repeats: int = 1,
    target_dims: List[int] = [50, 100, 200],
    imputation_method: str = 'mean',
    imputation_params: Dict[str, Any] = None,
    verbosity: int = 1,
    collect_predictions: bool = False,
    scoring: str = 'roc_auc',
    metrics: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Run nested CV evaluation with Representative Feature Generation (RFG).

    RFG performs BOTH embedding selection AND dimensionality selection in the inner CV.
    This extends run_multiple_evaluations_nlp() to select the best embedding TYPE
    (e.g., BoW Classic vs BoW TF-IDF vs Gemini) in addition to the best dimensionality
    for that embedding.

    Inner CV loop:
    1. For each embedding type, create dimensionality variants
    2. Select best (embedding_type, dimensionality) combination
    3. Use that combination on the outer test fold

    Args:
        embeddings_dict: Dictionary mapping embedding names to DataFrames
                        e.g., {'bow_classic': df1, 'bow_tfidf': df2, 'gemini': df3}
        baseline_df: Baseline features DataFrame
        y: Target labels
        model: Model to evaluate
        param_grid: Hyperparameter grid
        n_iterations: Number of evaluation iterations
        master_seed: Master seed for generating iteration seeds
        n_outer: Number of outer CV folds
        n_inner: Number of inner CV folds
        n_inner_repeats: Number of inner CV repeats
        target_dims: List of target dimensionalities for variants
        imputation_method: Imputation method ('mean', 'median', 'mice', 'svd', 'knn')
        imputation_params: Imputer parameters
        verbosity: Verbosity level

    Returns:
        Dictionary with aggregated results across all iterations
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

    metrics = normalize_metrics(metrics)

    if verbosity >= 1:
        print(f"\nRunning RFG evaluation with EMBEDDING + DIMENSIONALITY selection...")
        print(f"Embeddings: {list(embeddings_dict.keys())}")
        print(f"Target dimensions: {target_dims}")
        print(f"Iterations: {n_iterations}, Outer folds: {n_outer}, Inner folds: {n_inner}")

    # Create all embedding variants for all embeddings
    all_variants = {}
    for emb_name, emb_df in embeddings_dict.items():
        variants = create_embedding_variants(emb_df, emb_name, target_dims)
        all_variants.update(variants)

    if verbosity >= 1:
        print(f"Created {len(all_variants)} embedding variants total")

    # Storage for results across iterations
    all_auc_scores = []
    all_accuracy_scores = []
    all_aupr_scores = []
    all_f1_scores = []
    all_tp_counts = []
    all_fp_counts = []
    all_fn_counts = []
    all_fold_sizes = []
    all_best_variants = []
    all_thresholds = []

    # Initialize iteration predictions storage if requested
    if collect_predictions:
        iteration_predictions = {}

    # Generate seeds for each iteration
    rng = np.random.RandomState(master_seed)
    seeds = rng.randint(0, 10000, size=n_iterations)

    # Run evaluations
    for i, seed in enumerate(seeds):
        if verbosity >= 1:
            print(f"\nIteration {i + 1}/{n_iterations} (seed={seed})")

        # Create outer CV with iteration-specific seed
        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

        # Impute baseline features once per iteration (embeddings have no NaN)
        # Use iteration seed for imputation (happens once per iteration, not per fold)
        imputer_params = imputation_params if imputation_params else {}
        imputer = get_imputer(imputation_method, random_state=seed, **imputer_params)
        baseline_df_imputed = pd.DataFrame(
            imputer.fit_transform(baseline_df),
            columns=baseline_df.columns,
            index=baseline_df.index
        )

        # Storage for this iteration's folds
        iteration_auc_scores = []
        iteration_accuracy_scores = []
        iteration_aupr_scores = []
        iteration_f1_scores = []
        iteration_tp_counts = []
        iteration_fp_counts = []
        iteration_fn_counts = []
        iteration_fold_sizes = []
        iteration_best_variants = []
        iteration_thresholds = []

        # Initialize fold predictions storage for this iteration if requested
        if collect_predictions:
            fold_predictions = {}

        # Create RNG for generating fold-specific seeds
        fold_rng = np.random.RandomState(seed)

        # Outer CV loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(baseline_df_imputed, y)):
            if verbosity >= 2:
                print(f"  Outer fold {fold_idx + 1}/{n_outer}")

            # Generate fold-specific seed and derive component-specific seeds
            fold_seed = fold_rng.randint(0, 100000)
            component_rng = np.random.RandomState(fold_seed)
            inner_cv_seed = component_rng.randint(0, 100000)
            model_seed = component_rng.randint(0, 100000)

            # Create inner CV with fold-specific seed
            if n_inner_repeats > 1:
                inner_cv = RepeatedStratifiedKFold(
                    n_splits=n_inner,
                    n_repeats=n_inner_repeats,
                    random_state=inner_cv_seed
                )
            else:
                inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=inner_cv_seed)

            # Create scaler for this fold
            scaler = StandardScaler()

            # Inner CV: select best embedding variant (embedding type + dimensionality)
            (best_variant, test_auc, test_accuracy, test_aupr, test_f1,
             test_tp, test_fp, test_fn, fold_size, feature_importance, y_true, y_pred_proba, best_params, threshold_selected) = inner_cv_nlp(
                embedding_variants=all_variants,  # All embeddings + all variants
                baseline_df=baseline_df_imputed,
                y=y,
                model=model,
                param_grid=param_grid,
                inner_cv=inner_cv,
                scaler=scaler,
                train_idx=train_idx,
                test_idx=test_idx,
                random_seed=model_seed,
                scoring=scoring
            )

            if verbosity >= 1:
                # Format params nicely
                params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
                print(f"  Outer fold {fold_idx + 1}/{n_outer}")
                print(f"    Selected variant: {best_variant}")
                print(f"    Selected params: {params_str}")
                print(f"    Fold AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")

            # Store fold results
            iteration_auc_scores.append(test_auc)
            iteration_accuracy_scores.append(test_accuracy)
            iteration_aupr_scores.append(test_aupr)
            iteration_f1_scores.append(test_f1)
            iteration_tp_counts.append(test_tp)
            iteration_fp_counts.append(test_fp)
            iteration_fn_counts.append(test_fn)
            iteration_fold_sizes.append(fold_size)
            iteration_best_variants.append(best_variant)
            if scoring == 'f1':
                iteration_thresholds.append(threshold_selected)

            # Store predictions if requested
            if collect_predictions:
                fold_predictions[fold_idx] = {
                    'y_true': y_true.copy(),
                    'y_pred': y_pred_proba.copy(),
                    'test_idx': test_idx.copy()
                }

        # Store iteration results
        all_auc_scores.append(np.mean(iteration_auc_scores))
        all_accuracy_scores.append(np.mean(iteration_accuracy_scores))
        all_aupr_scores.append(np.mean(iteration_aupr_scores))
        all_f1_scores.append(np.mean(iteration_f1_scores))
        all_tp_counts.extend(iteration_tp_counts)
        all_fp_counts.extend(iteration_fp_counts)
        all_fn_counts.extend(iteration_fn_counts)
        all_fold_sizes.extend(iteration_fold_sizes)
        all_best_variants.extend(iteration_best_variants)
        if scoring == 'f1':
            all_thresholds.extend(iteration_thresholds)

        # Store fold predictions for this iteration
        if collect_predictions:
            iteration_predictions[f'iteration_{i}'] = fold_predictions

        # Log selected variants for this iteration
        if verbosity >= 1:
            from collections import Counter
            variant_counts = Counter(iteration_best_variants)
            variants_summary = ", ".join([f"{v}({c})" for v, c in variant_counts.items()])
            print(f"  Selected variants: {variants_summary}")
            metric_lists = {
                'auc': iteration_auc_scores,
                'accuracy': iteration_accuracy_scores,
                'aupr': iteration_aupr_scores,
                'f1': iteration_f1_scores
            }
            metric_summary = [f"{metric.upper()}: {np.mean(metric_lists[metric]):.4f}" for metric in metrics]
            print(f"  Mean {'; '.join(metric_summary)}")

    # Calculate summary statistics
    results = {
        'auc_scores': all_auc_scores,
        'accuracy_scores': all_accuracy_scores,
        'aupr_scores': all_aupr_scores,
        'f1_scores': all_f1_scores,
        'tp_counts': all_tp_counts,
        'fp_counts': all_fp_counts,
        'fn_counts': all_fn_counts,
        'fold_sizes': all_fold_sizes,
        'best_variants': all_best_variants,
        'mean_auc': np.mean(all_auc_scores),
        'std_auc': np.std(all_auc_scores),
        'mean_accuracy': np.mean(all_accuracy_scores),
        'std_accuracy': np.std(all_accuracy_scores),
        'mean_aupr': np.mean(all_aupr_scores),
        'std_aupr': np.std(all_aupr_scores),
        'mean_f1': np.mean(all_f1_scores),
        'std_f1': np.std(all_f1_scores),
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
        'seeds': seeds.tolist()  # Add seeds for reproducibility
    }

    if scoring == 'f1' and all_thresholds:
        results['thresholds'] = all_thresholds

    # Add iteration predictions if requested
    if collect_predictions:
        results['iteration_predictions'] = iteration_predictions

    # Print final summary
    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS (RFG)")
        print(f"{'='*80}")
        for metric in metrics:
            mean_key = f"mean_{metric}"
            std_key = f"std_{metric}"
            if mean_key in results and std_key in results:
                print(f"  Mean {metric.upper()}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")

        # Show most common variants
        from collections import Counter
        variant_counts = Counter(all_best_variants)
        print(f"\n  Most frequently selected variants:")
        for variant, count in variant_counts.most_common(5):
            percentage = (count / len(all_best_variants)) * 100
            print(f"    {variant}: {count}/{len(all_best_variants)} ({percentage:.1f}%)")
        print(f"{'='*80}")

    return results
