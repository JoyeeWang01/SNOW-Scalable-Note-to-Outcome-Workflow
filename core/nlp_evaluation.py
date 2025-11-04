"""
NLP Feature Evaluation with Embedding and Dimensionality Selection.

This module implements nested cross-validation for NLP features (BoW, TF-IDF, embeddings)
with automatic selection of best embedding type and dimensionality during inner CV.

Based on evaluation_code_0701.ipynb nested_cv_evaluation_nlp function.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score

# Import custom logging print function
from core.log_utils import print


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
    random_seed: int = None
) -> Tuple[str, float, float, Any, np.ndarray, np.ndarray]:
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
        Tuple of (best_variant_name, test_auc, test_accuracy, feature_importance, y_true, y_pred_proba, best_params)
    """
    best_overall_score = -1
    best_model = None
    best_variant_name = None
    best_X_test_processed = None
    best_cv_result = None
    best_params = None

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

        grid_search = GridSearchCV(
            estimator=model_clone,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
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

    # Evaluate best model on test set
    y_pred_proba = best_model.predict_proba(best_X_test_processed)[:, 1]
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

    # Return predictions along with other results
    return best_variant_name, test_auc, test_accuracy, feature_importance, y_test.values, y_pred_proba, best_params


def inner_cv_nlp_model_selection(
    embedding_variants: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
    y: pd.Series,
    models_config: Dict[str, Dict],
    inner_cv,
    scaler: StandardScaler,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_seed: int = None
) -> Tuple[str, str, float, float, Any, np.ndarray, np.ndarray, Dict[str, Any]]:
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
                 feature_importance, y_true, y_pred_proba, best_params)
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

            # Grid search for hyperparameter tuning
            param_grid = model_config['param_grid']
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1,
                error_score='raise'
            )

            try:
                grid_search.fit(X_train_processed, y_train)
                score = grid_search.best_score_
                model_scores[variant_name] = score

                # Update best if this combination is better
                if score > best_overall_score:
                    best_overall_score = score
                    best_model_instance = grid_search.best_estimator_
                    best_model_name = model_name
                    best_variant_name = variant_name
                    best_X_test_processed = X_test_processed
                    best_params = grid_search.best_params_

            except Exception as e:
                model_scores[variant_name] = np.nan
                continue

        all_scores[model_name] = model_scores

    if best_model_instance is None:
        raise RuntimeError("All model/variant combinations failed during inner CV")

    # Evaluate best model on test set
    y_pred = best_model_instance.predict(best_X_test_processed)
    y_pred_proba = best_model_instance.predict_proba(best_X_test_processed)[:, 1]

    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_accuracy = accuracy_score(y_test, y_pred)

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
            feature_importance, y_test.values, y_pred_proba, best_params)


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
    collect_predictions: bool = False  # Not currently used (NLP features excluded from permutation tests)
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
        imputation_method: Imputation method ('mean', 'mice', 'svd', 'knn')
        imputation_params: Imputer parameters
        verbosity: Verbosity level
        collect_predictions: Whether to collect predictions (not used; NLP excluded from permutation tests)

    Returns:
        Dictionary with aggregated results across all iterations
    """
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
    all_auc_scores = []
    all_accuracy_scores = []
    all_best_variants = []

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
        from core.evaluation import get_imputer
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
        iteration_best_variants = []

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
            best_variant, test_auc, test_accuracy, feature_importance, y_true, y_pred_proba, best_params = inner_cv_nlp(
                embedding_variants=embedding_variants,
                baseline_df=baseline_df_imputed,
                y=y,
                model=model,
                param_grid=param_grid,
                inner_cv=inner_cv,
                scaler=scaler,
                train_idx=train_idx,
                test_idx=test_idx,
                random_seed=model_seed
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
            iteration_best_variants.append(best_variant)

            # Store predictions if requested
            if collect_predictions:
                fold_predictions[fold_idx] = {
                    'y_true': y_true.copy(),
                    'y_pred': y_pred_proba.copy(),
                    'test_idx': test_idx.copy()
                }

        # Store iteration results (mean across folds for this iteration)
        all_auc_scores.append(np.mean(iteration_auc_scores))
        all_accuracy_scores.append(np.mean(iteration_accuracy_scores))
        all_best_variants.extend(iteration_best_variants)

        # Store fold predictions for this iteration
        if collect_predictions:
            iteration_predictions[f'iteration_{i}'] = fold_predictions

        # Log selected variants for this iteration
        if verbosity >= 1:
            from collections import Counter
            variant_counts = Counter(iteration_best_variants)
            variants_summary = ", ".join([f"{v}({c})" for v, c in variant_counts.items()])
            print(f"  Selected variants: {variants_summary}")
            print(f"  Mean AUC: {np.mean(iteration_auc_scores):.4f}, Mean Accuracy: {np.mean(iteration_accuracy_scores):.4f}")

    # Calculate summary statistics (same format as standard evaluation)
    results = {
        'auc_scores': all_auc_scores,
        'accuracy_scores': all_accuracy_scores,
        'best_variants': all_best_variants,
        'mean_auc': np.mean(all_auc_scores),
        'std_auc': np.std(all_auc_scores),
        'mean_accuracy': np.mean(all_accuracy_scores),
        'std_accuracy': np.std(all_accuracy_scores),
        'seeds': seeds.tolist()  # Add seeds for reproducibility
    }

    # Add iteration predictions if requested
    if collect_predictions:
        results['iteration_predictions'] = iteration_predictions

    # Print final summary
    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"  Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

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
    collect_predictions: bool = False
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
        imputation_method: Imputation method ('mean', 'mice', 'svd', 'knn')
        imputation_params: Additional imputation parameters
        verbosity: Verbosity level
        collect_predictions: Whether to collect predictions

    Returns:
        Dictionary with results including model selection statistics
    """
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from core.ml_models import SVDImputer
    from collections import Counter

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
    all_best_models = []
    all_best_variants = []
    all_best_params = []

    if collect_predictions:
        iteration_predictions = {}

    # Outer CV iterations
    for i in range(n_iterations):
        iteration_seed = None if master_seed is None else master_seed + i * 1000

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
        if imputation_method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif imputation_method == 'mice':
            imputer = IterativeImputer(
                max_iter=imputation_params.get('max_iter', 10),
                random_state=iteration_seed
            )
        elif imputation_method == 'svd':
            imputer = SVDImputer(
                n_components=imputation_params.get('n_components', 3),
                max_iter=imputation_params.get('max_iter', 500),
                tol=imputation_params.get('tol', 1e-4),
                random_state=iteration_seed
            )
        elif imputation_method == 'knn':
            imputer = KNNImputer(n_neighbors=imputation_params.get('n_neighbors', 5))
        else:
            raise ValueError(f"Unknown imputation method: {imputation_method}")

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
             feature_importance, y_true, y_pred_proba, best_params) = inner_cv_nlp_model_selection(
                embedding_variants=embedding_variants,
                baseline_df=baseline_imputed,
                y=y,
                models_config=models_config,
                inner_cv=inner_cv,
                scaler=scaler,
                train_idx=train_idx,
                test_idx=test_idx,
                random_seed=fold_seed
            )

            all_auc_scores.append(test_auc)
            all_accuracy_scores.append(test_accuracy)
            all_best_models.append(best_model_name)
            all_best_variants.append(best_variant_name)
            all_best_params.append(best_params)

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
            print(f"  Iteration {i+1} mean AUC: {np.mean(all_auc_scores[i*n_outer:(i+1)*n_outer]):.4f}\n")

    # Aggregate results
    model_counts = Counter(all_best_models)
    variant_counts = Counter(all_best_variants)

    results = {
        'all_auc_scores': all_auc_scores,
        'all_accuracy_scores': all_accuracy_scores,
        'mean_auc': np.mean(all_auc_scores),
        'std_auc': np.std(all_auc_scores),
        'mean_accuracy': np.mean(all_accuracy_scores),
        'std_accuracy': np.std(all_accuracy_scores),
        'selected_models': all_best_models,
        'selected_variants': all_best_variants,
        'selected_params': all_best_params,
        'model_selection_counts': model_counts,
        'variant_selection_counts': variant_counts,
        'seeds': seeds.tolist()  # Add seeds for reproducibility
    }

    if collect_predictions:
        results['iteration_predictions'] = iteration_predictions

    # Print summary
    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"MODEL SELECTION SUMMARY - {embedding_name}")
        print(f"{'='*80}")
        print(f"Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
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
    collect_predictions: bool = False
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
        imputation_method: Imputation method ('mean', 'mice', 'svd', 'knn')
        imputation_params: Imputer parameters
        verbosity: Verbosity level

    Returns:
        Dictionary with aggregated results across all iterations
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

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
    all_best_variants = []

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
        from core.evaluation import get_imputer
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
        iteration_best_variants = []

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
            best_variant, test_auc, test_accuracy, feature_importance, y_true, y_pred_proba, best_params = inner_cv_nlp(
                embedding_variants=all_variants,  # All embeddings + all variants
                baseline_df=baseline_df_imputed,
                y=y,
                model=model,
                param_grid=param_grid,
                inner_cv=inner_cv,
                scaler=scaler,
                train_idx=train_idx,
                test_idx=test_idx,
                random_seed=model_seed
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
            iteration_best_variants.append(best_variant)

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
        all_best_variants.extend(iteration_best_variants)

        # Store fold predictions for this iteration
        if collect_predictions:
            iteration_predictions[f'iteration_{i}'] = fold_predictions

        # Log selected variants for this iteration
        if verbosity >= 1:
            from collections import Counter
            variant_counts = Counter(iteration_best_variants)
            variants_summary = ", ".join([f"{v}({c})" for v, c in variant_counts.items()])
            print(f"  Selected variants: {variants_summary}")
            print(f"  Mean AUC: {np.mean(iteration_auc_scores):.4f}, Mean Accuracy: {np.mean(iteration_accuracy_scores):.4f}")

    # Calculate summary statistics
    results = {
        'auc_scores': all_auc_scores,
        'accuracy_scores': all_accuracy_scores,
        'best_variants': all_best_variants,
        'mean_auc': np.mean(all_auc_scores),
        'std_auc': np.std(all_auc_scores),
        'mean_accuracy': np.mean(all_accuracy_scores),
        'std_accuracy': np.std(all_accuracy_scores),
        'seeds': seeds.tolist()  # Add seeds for reproducibility
    }

    # Add iteration predictions if requested
    if collect_predictions:
        results['iteration_predictions'] = iteration_predictions

    # Print final summary
    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS (RFG)")
        print(f"{'='*80}")
        print(f"  Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

        # Show most common variants
        from collections import Counter
        variant_counts = Counter(all_best_variants)
        print(f"\n  Most frequently selected variants:")
        for variant, count in variant_counts.most_common(5):
            percentage = (count / len(all_best_variants)) * 100
            print(f"    {variant}: {count}/{len(all_best_variants)} ({percentage:.1f}%)")
        print(f"{'='*80}")

    return results
