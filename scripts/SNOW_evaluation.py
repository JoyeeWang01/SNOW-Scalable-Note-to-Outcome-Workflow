"""
Evaluate SNOW-generated features and embeddings using nested cross-validation.

This script:
1. Loads baseline features, SNOW features, and embeddings (BoW, Gemini)
2. Evaluates multiple feature combinations using nested CV
3. Runs 100 iterations with different random seeds for statistical robustness
4. Generates comparison plots for each machine learning model
5. Saves detailed results and summary tables

Usage:
    cd stanford/scripts
    python SNOW_oncology_evaluation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.log_utils import setup_logging, print
setup_logging("evaluation")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Import modular components
from config.evaluation_config import (
    STRUCTURED_FILE_PATH,
    LABEL_COL,
    INDEX_COL,
    NOTES_COL,
    SNOW_FEATURES_PATH,
    EMBEDDING_FILES,
    ADDITIONAL_FEATURES,
    N_ITERATIONS,
    MASTER_SEED,
    N_OUTER_FOLDS,
    N_INNER_FOLDS,
    INNER_CV_REPEATS,
    TARGET_DIMS,
    IMPUTATION_METHOD,
    MICE_MAX_ITER,
    SVD_N_COMPONENTS,
    SVD_MAX_ITER,
    SVD_TOL,
    KNN_N_NEIGHBORS,
    MODELS,
    FEATURE_SETS,
    RESULTS_DIR,
    SAVE_DETAILED_RESULTS,
    PLOT_FIGSIZE,
    PLOT_DPI,
    VERBOSITY,
    ENABLE_PERMUTATION_TESTS,
    PERMUTATION_TEST_PAIRS,
    PERMUTATION_TEST_MODELS,
    PERMUTATION_N_PERMUTATIONS,
    PERMUTATION_SEED
)

from core.data_loader import load_all_features, combine_feature_sets
from core.evaluation import run_multiple_evaluations
from core.nlp_evaluation import run_multiple_evaluations_nlp, run_multiple_evaluations_rfg
from core.ml_models import RandomFeatureModel
from core.visualization import (
    plot_model_comparison,
    save_results_table,
    save_detailed_results as save_detailed_csv,
    save_comprehensive_results,
    save_permutation_results
)
from core.permutation_test import run_permutation_test_pipeline

def get_model_instance(model_config: dict, random_state: int = None, model_name: str = None):
    """
    Get model instance from configuration.

    Args:
        model_config: Model configuration dict
        random_state: Random seed
        model_name: Name of the model (for custom models)

    Returns:
        Model instance
    """
    if model_config['class'] == 'custom':
        # Legacy support for custom models
        if model_name == 'RandomFeature':
            return RandomFeatureModel(random_state=random_state)
        else:
            raise ValueError(f"Unknown custom model: {model_name}")
    else:
        # Import model class dynamically
        module_path, class_name = model_config['class'].rsplit('.', 1)

        # Handle core modules (relative imports from this package)
        if module_path.startswith('core.'):
            # Import from the core package
            import sys
            from pathlib import Path
            # The core module should already be importable since we added it to sys.path
            module = __import__(module_path, fromlist=[class_name])
        else:
            # Standard sklearn or other third-party imports
            module = __import__(module_path, fromlist=[class_name])

        model_class = getattr(module, class_name)

        # Merge params with random_state
        params = model_config['params'].copy()
        if random_state is not None:
            params['random_state'] = random_state

        return model_class(**params)


def is_nlp_feature_set(sources: List[str]) -> Tuple[bool, str]:
    """
    Check if feature set contains exactly one NLP embedding (for dimensionality selection).

    Args:
        sources: List of feature source names

    Returns:
        Tuple of (is_nlp, embedding_name) where is_nlp is True if feature set
        contains baseline + exactly one NLP embedding
    """
    nlp_embeddings = ['bow_classic', 'bow_tfidf', 'gemini']

    # Count NLP embeddings in sources
    nlp_sources = [s for s in sources if s in nlp_embeddings]

    # Must have: baseline (structured) + exactly one NLP embedding
    has_baseline = 'structured' in sources
    has_single_nlp = len(nlp_sources) == 1
    has_no_other_sources = len(sources) == 2  # Only baseline + one NLP

    if has_baseline and has_single_nlp and has_no_other_sources:
        return True, nlp_sources[0]

    return False, None


def is_rfg_feature_set(sources: List[str]) -> bool:
    """
    Check if feature set uses RFG (Representative Feature Generation).

    RFG selects the best embedding type AND dimensionality in the inner CV.

    Args:
        sources: List of feature source names

    Returns:
        True if this is an RFG feature set (baseline + 'rfg')
    """
    return 'structured' in sources and 'rfg' in sources


def main():
    """Main evaluation function."""
    print("="*80)
    print("SNOW ONCOLOGY FEATURE EVALUATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_results_dir = os.path.join(RESULTS_DIR, f"evaluation_{timestamp}")
    os.makedirs(run_results_dir, exist_ok=True)

    # Reconfigure logging to save to the evaluation folder
    setup_logging("evaluation", log_dir=run_results_dir)
    print(f"Results directory: {run_results_dir}\n")
    print(f"Log file saved to: {run_results_dir}\n")

    # ========================================================================
    # Load Data
    # ========================================================================
    print("="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)

    all_data = load_all_features(
        structured_path=STRUCTURED_FILE_PATH,
        label_col=LABEL_COL,
        index_col=INDEX_COL,  # Validate row alignment if INDEX_COL is not None
        notes_col=NOTES_COL,  # Remove notes column from all feature files
        snow_path=SNOW_FEATURES_PATH if os.path.exists(SNOW_FEATURES_PATH) else None,
        embedding_paths=EMBEDDING_FILES,  # Load embeddings from dictionary
        additional_paths=ADDITIONAL_FEATURES  # Load additional feature sets
    )

    y = all_data['labels'].values

    # Print column information for each feature source
    print("\n" + "="*80)
    print("FEATURE COLUMNS SUMMARY")
    print("="*80)

    print(f"\nBaseline features ({all_data['structured'].shape[1]} columns):")
    print(f"  {list(all_data['structured'].columns)}")

    if 'snow' in all_data:
        print(f"\nSNOW features ({all_data['snow'].shape[1]} columns):")
        print(f"  {list(all_data['snow'].columns)}")

    # Print additional feature sets
    for add_name in ADDITIONAL_FEATURES.keys():
        if add_name in all_data:
            print(f"\n{add_name} ({all_data[add_name].shape[1]} columns):")
            print(f"  {list(all_data[add_name].columns)}")

    print("="*80 + "\n")

    # ========================================================================
    # Dynamic Feature Set Generation
    # ========================================================================
    print("="*80)
    print("STEP 2: DYNAMIC FEATURE SET GENERATION")
    print("="*80)

    # Start with base FEATURE_SETS from config
    # Use only manually defined feature sets from config (no auto-generation)
    dynamic_feature_sets = FEATURE_SETS.copy()

    print(f"Total feature sets to evaluate: {len(dynamic_feature_sets)}")
    for name, config in dynamic_feature_sets.items():
        print(f"  - {name}: {config['description']}")
    print("="*80 + "\n")

    # ========================================================================
    # Setup Imputation Parameters
    # ========================================================================
    imputation_params = {}
    if IMPUTATION_METHOD == 'mice':
        imputation_params = {'max_iter': MICE_MAX_ITER}
    elif IMPUTATION_METHOD == 'svd':
        imputation_params = {
            'n_components': SVD_N_COMPONENTS,
            'max_iter': SVD_MAX_ITER,
            'tol': SVD_TOL
        }
    elif IMPUTATION_METHOD == 'knn':
        imputation_params = {'n_neighbors': KNN_N_NEIGHBORS}

    # ========================================================================
    # Evaluate Each Model (or Run Model Selection)
    # ========================================================================
    all_model_results = {}

    # Evaluate each model separately
    total_models = len(MODELS)
    for model_idx, (model_name, model_config) in enumerate(MODELS.items(), 1):
            print("\n" + "="*80)
            print(f"EVALUATING MODEL {model_idx}/{total_models}: {model_name}")
            print("="*80)

            # Store results for this model across all feature sets
            model_feature_results = {}

            # Evaluate each feature set (using dynamic_feature_sets)
            for feature_set_name, feature_set_config in dynamic_feature_sets.items():
                print(f"\n{'='*80}")
                print(f"Feature Set: {feature_set_name}")
                print(f"Description: {feature_set_config['description']}")
                print(f"{'='*80}")

                # Check if this is an NLP feature set (baseline + single NLP embedding)
                is_nlp, embedding_name = is_nlp_feature_set(feature_set_config['sources'])
                is_rfg = is_rfg_feature_set(feature_set_config['sources'])

                if is_rfg:
                    # Use RFG evaluation with BOTH embedding AND dimensionality selection
                    print(f"Using RFG (Representative Feature Generation) evaluation")
                    print(f"Selecting best embedding + dimensionality in inner CV")

                    # Collect all available embeddings from EMBEDDING_FILES
                    embeddings_dict = {}
                    for emb_name in EMBEDDING_FILES.keys():
                        if emb_name in all_data:
                            embeddings_dict[emb_name] = all_data[emb_name]

                    if len(embeddings_dict) == 0:
                        print(f"⚠️  Skipping {feature_set_name}: No NLP embeddings available")
                        continue

                    print(f"Candidate embeddings from EMBEDDING_FILES: {list(EMBEDDING_FILES.keys())}")
                    print(f"Available embeddings: {list(embeddings_dict.keys())}")

                    # Get baseline features
                    baseline_df = all_data['structured']

                    # Get model instance
                    model = get_model_instance(model_config, random_state=None, model_name=model_name)

                    # Run RFG evaluation with embedding + dimensionality selection
                    results = run_multiple_evaluations_rfg(
                        embeddings_dict=embeddings_dict,
                        baseline_df=baseline_df,
                        y=pd.Series(y, index=baseline_df.index),
                        model=model,
                        param_grid=model_config['param_grid'],
                        n_iterations=N_ITERATIONS,
                        master_seed=MASTER_SEED,
                        n_outer=N_OUTER_FOLDS,
                        n_inner=N_INNER_FOLDS,
                        n_inner_repeats=INNER_CV_REPEATS,
                        target_dims=TARGET_DIMS,
                        imputation_method=IMPUTATION_METHOD,
                        imputation_params=imputation_params,
                        verbosity=VERBOSITY,
                        collect_predictions=ENABLE_PERMUTATION_TESTS  # Collect predictions for permutation tests
                    )

                elif is_nlp:
                    # Use NLP evaluation with dimensionality selection
                    print(f"Using NLP evaluation with dimensionality selection for {embedding_name}")

                    # Get embedding DataFrame
                    embedding_df = all_data.get(embedding_name)
                    if embedding_df is None:
                        print(f"⚠️  Skipping {feature_set_name}: {embedding_name} not available")
                        continue

                    # Get baseline features
                    baseline_df = all_data['structured']

                    # Get model instance
                    model = get_model_instance(model_config, random_state=None, model_name=model_name)

                    # Run NLP evaluation with dimensionality selection
                    # Same parameters as standard evaluation, but with dimension selection in inner CV
                    results = run_multiple_evaluations_nlp(
                        embedding_df=embedding_df,
                        embedding_name=embedding_name,
                        baseline_df=baseline_df,
                        y=pd.Series(y, index=baseline_df.index),
                        model=model,
                        param_grid=model_config['param_grid'],
                        n_iterations=N_ITERATIONS,
                        master_seed=MASTER_SEED,
                        n_outer=N_OUTER_FOLDS,
                        n_inner=N_INNER_FOLDS,
                        n_inner_repeats=INNER_CV_REPEATS,
                        target_dims=TARGET_DIMS,
                        imputation_method=IMPUTATION_METHOD,
                        imputation_params=imputation_params,
                        verbosity=VERBOSITY,
                        collect_predictions=ENABLE_PERMUTATION_TESTS  # Collect predictions for permutation tests
                    )

                else:
                    # Standard evaluation (no dimensionality selection)
                    # Combine features according to sources
                    # Collect additional feature DataFrames
                    additional_dfs = {name: all_data[name] for name in ADDITIONAL_FEATURES.keys() if name in all_data}

                    # Collect embedding DataFrames
                    embeddings_dfs = {name: all_data[name] for name in EMBEDDING_FILES.keys() if name in all_data}

                    try:
                        X_combined = combine_feature_sets(
                            sources=feature_set_config['sources'],
                            structured_df=all_data['structured'],
                            snow_df=all_data.get('snow'),
                            embeddings_dfs=embeddings_dfs,
                            additional_dfs=additional_dfs
                        )
                    except (ValueError, KeyError) as e:
                        print(f"⚠️  Skipping {feature_set_name}: {e}")
                        continue

                    print(f"Combined feature matrix: {X_combined.shape}")

                    # Ensure all columns are numeric before converting to numpy
                    non_numeric_cols = X_combined.select_dtypes(exclude=[np.number]).columns.tolist()
                    if non_numeric_cols:
                        print(f"⚠️  WARNING: Found non-numeric columns that should have been removed: {non_numeric_cols}")
                        print(f"    Removing these columns to proceed...")
                        X_combined = X_combined.select_dtypes(include=[np.number])
                        print(f"    New shape after removing non-numeric columns: {X_combined.shape}")

                    # Convert to numpy
                    X = X_combined.values

                    # Get model instance (without random state, will be set per iteration)
                    model = get_model_instance(model_config, random_state=None, model_name=model_name)

                    # Prepare feature names for tracking (for all non-NLP feature sets)
                    # NLP feature sets use is_nlp_feature_set and have their own importance tracking
                    feature_names = list(X_combined.columns)

                    # Run multiple evaluations
                    # Collect predictions if permutation tests are enabled
                    results = run_multiple_evaluations(
                        X=X,
                        y=y,
                        model=model,
                        param_grid=model_config['param_grid'],
                        n_iterations=N_ITERATIONS,
                        master_seed=MASTER_SEED,
                        n_outer=N_OUTER_FOLDS,
                        n_inner=N_INNER_FOLDS,
                        n_inner_repeats=INNER_CV_REPEATS,
                        imputation_method=IMPUTATION_METHOD,
                        imputation_params=imputation_params,
                        verbosity=VERBOSITY,
                        feature_names=feature_names,
                        collect_predictions=ENABLE_PERMUTATION_TESTS  # Collect predictions for permutation tests
                    )

                # Store results
                model_feature_results[feature_set_name] = results

            # Store results for this model
            all_model_results[model_name] = model_feature_results

            # Save model-specific results
            print(f"\n{'='*80}")
            print(f"SAVING RESULTS FOR {model_name}")
            print(f"{'='*80}")
            model_results_dir = os.path.join(run_results_dir, model_name)
            os.makedirs(model_results_dir, exist_ok=True)

            # Save summary table
            save_results_table(
                results_dict=model_feature_results,
                output_path=os.path.join(model_results_dir, f"{model_name}_summary.csv"),
                metrics=['auc', 'accuracy']
            )

            # Save comprehensive results for reproducibility (always enabled)
            print(f"\nSaving comprehensive results for {model_name}...")
            save_comprehensive_results(
                results_dict=model_feature_results,
                output_dir=model_results_dir,
                model_name=model_name
            )

            # Save detailed results if enabled (legacy format)
            if SAVE_DETAILED_RESULTS:
                save_detailed_csv(
                    results_dict=model_feature_results,
                    output_path=os.path.join(model_results_dir, f"{model_name}_detailed.csv")
                )

            # Save feature importance for all feature sets (models with feature_importances_ or coef_)
            print(f"\nSaving feature importance for {model_name}...")
            for feature_set_name, results in model_feature_results.items():
                if 'feature_importance' in results:
                    importance_df = results['feature_importance']
                    # Create safe filename from feature set name
                    safe_name = feature_set_name.replace(' ', '_').replace('+', '').replace('/', '_')
                    importance_path = os.path.join(model_results_dir, f"{model_name}_{safe_name}_feature_importance.csv")
                    importance_df.to_csv(importance_path, index=False)
                    print(f"  ✓ Saved {feature_set_name}: {importance_path}")
                    print(f"    Top 5 features: {', '.join(importance_df.head(5)['feature'].tolist())}")

            print(f"\n✓ Completed evaluation for model {model_idx}/{total_models}: {model_name}")

    # ========================================================================
    # Generate Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    for model_name in all_model_results.keys():
        model_results_dir = os.path.join(run_results_dir, model_name)

        # Generate plots for AUC
        print(f"\nGenerating AUC plots for {model_name}...")
        plot_model_comparison(
            model_results={model_name: all_model_results[model_name]},
            metric='auc',
            figsize=PLOT_FIGSIZE,
            output_dir=model_results_dir,
            dpi=PLOT_DPI
        )

        # Generate plots for Accuracy
        print(f"Generating Accuracy plots for {model_name}...")
        plot_model_comparison(
            model_results={model_name: all_model_results[model_name]},
            metric='accuracy',
            figsize=PLOT_FIGSIZE,
            output_dir=model_results_dir,
            dpi=PLOT_DPI
        )

    # ========================================================================
    # Permutation Tests (if enabled)
    # ========================================================================
    if ENABLE_PERMUTATION_TESTS and len(PERMUTATION_TEST_PAIRS) > 0:
        print("\n" + "="*80)
        print("RUNNING PERMUTATION TESTS")
        print("="*80)
        print(f"Number of test pairs: {len(PERMUTATION_TEST_PAIRS)}")
        print(f"Permutation iterations: {PERMUTATION_N_PERMUTATIONS}")
        print(f"Test pairs:")
        for fs1, fs2 in PERMUTATION_TEST_PAIRS:
            print(f"  - {fs2} vs {fs1}")
        print("="*80)

        permutation_results = {}

        for model_name, model_config in MODELS.items():
            # Skip models not in PERMUTATION_TEST_MODELS (if specified)
            if PERMUTATION_TEST_MODELS is not None and model_name not in PERMUTATION_TEST_MODELS:
                print(f"\n⊘ Skipping permutation tests for {model_name} (not in PERMUTATION_TEST_MODELS)")
                continue

            print(f"\n{'='*80}")
            print(f"Permutation Tests for {model_name}")
            print(f"{'='*80}")

            model_perm_results = {}

            # Loop through all configured test pairs
            for fs1_name, fs2_name in PERMUTATION_TEST_PAIRS:
                print(f"\nTesting: {fs2_name} vs {fs1_name}")

                # Validate that both feature sets exist
                if fs1_name not in dynamic_feature_sets:
                    print(f"  ⚠️  Skipping: '{fs1_name}' not found in FEATURE_SETS")
                    continue
                if fs2_name not in dynamic_feature_sets:
                    print(f"  ⚠️  Skipping: '{fs2_name}' not found in FEATURE_SETS")
                    continue

                # Skip RFG feature sets (dimensionality AND embedding selection makes permutation testing complex)
                is_rfg1 = is_rfg_feature_set(dynamic_feature_sets[fs1_name]['sources'])
                is_rfg2 = is_rfg_feature_set(dynamic_feature_sets[fs2_name]['sources'])
                if is_rfg1 or is_rfg2:
                    print(f"  ⚠️  Skipping: RFG feature sets not supported for permutation tests")
                    continue

                # Get feature matrices
                additional_dfs = {name: all_data[name] for name in ADDITIONAL_FEATURES.keys() if name in all_data}

                # Collect embedding DataFrames
                embeddings_dfs = {name: all_data[name] for name in EMBEDDING_FILES.keys() if name in all_data}

                try:
                    X1_combined = combine_feature_sets(
                        sources=dynamic_feature_sets[fs1_name]['sources'],
                        structured_df=all_data['structured'],
                        snow_df=all_data.get('snow'),
                        embeddings_dfs=embeddings_dfs,
                        additional_dfs=additional_dfs
                    )

                    X2_combined = combine_feature_sets(
                        sources=dynamic_feature_sets[fs2_name]['sources'],
                        structured_df=all_data['structured'],
                        snow_df=all_data.get('snow'),
                        embeddings_dfs=embeddings_dfs,
                        additional_dfs=additional_dfs
                    )
                except (ValueError, KeyError) as e:
                    print(f"  ⚠️  Skipping: Error combining features - {e}")
                    continue

                # Get model instances
                model1 = get_model_instance(model_config, random_state=None, model_name=model_name)
                model2 = get_model_instance(model_config, random_state=None, model_name=model_name)

                # Get pre-computed predictions and AUC stats if available
                predictions_1 = None
                predictions_2 = None
                auc_stats_1 = None
                auc_stats_2 = None

                if model_name in all_model_results:
                    if fs1_name in all_model_results[model_name]:
                        fs1_results = all_model_results[model_name][fs1_name]
                        predictions_1 = fs1_results.get('iteration_predictions')
                        auc_stats_1 = {
                            'mean_auc': fs1_results.get('mean_auc'),
                            'std_auc': fs1_results.get('std_auc')
                        }
                    if fs2_name in all_model_results[model_name]:
                        fs2_results = all_model_results[model_name][fs2_name]
                        predictions_2 = fs2_results.get('iteration_predictions')
                        auc_stats_2 = {
                            'mean_auc': fs2_results.get('mean_auc'),
                            'std_auc': fs2_results.get('std_auc')
                        }

                # Run permutation test (will use pre-computed predictions if available)
                perm_result = run_permutation_test_pipeline(
                    X1=X1_combined.values,
                    X2=X2_combined.values,
                    y=y,
                    model1=model1,
                    model2=model2,
                    param_grid1=model_config['param_grid'],
                    param_grid2=model_config['param_grid'],
                    n_iterations=N_ITERATIONS,
                    master_seed=MASTER_SEED,
                    n_outer=N_OUTER_FOLDS,
                    n_inner=N_INNER_FOLDS,
                    n_inner_repeats=INNER_CV_REPEATS,
                    imputation_method=IMPUTATION_METHOD,
                    imputation_params=imputation_params,
                    feature_set_1_name=fs1_name,
                    feature_set_2_name=fs2_name,
                    n_permutations=PERMUTATION_N_PERMUTATIONS,
                    perm_seed=PERMUTATION_SEED,
                    verbosity=VERBOSITY,
                    predictions_1=predictions_1,  # Pass pre-computed predictions
                    predictions_2=predictions_2,  # Pass pre-computed predictions
                    auc_stats_1=auc_stats_1,      # Pass AUC statistics
                    auc_stats_2=auc_stats_2       # Pass AUC statistics
                )

                test_key = f"{fs2_name} vs {fs1_name}"
                model_perm_results[test_key] = perm_result

            permutation_results[model_name] = model_perm_results

            # Save permutation results for this model
            model_results_dir = os.path.join(run_results_dir, model_name)
            save_permutation_results(model_perm_results, model_results_dir, model_name)

        print("\n" + "="*80)
        print("PERMUTATION TESTS COMPLETE")
        print("="*80)

    # ========================================================================
    # Save Configuration
    # ========================================================================
    config_summary = {
        'timestamp': timestamp,
        'n_iterations': N_ITERATIONS,
        'master_seed': MASTER_SEED,
        'n_outer_folds': N_OUTER_FOLDS,
        'n_inner_folds': N_INNER_FOLDS,
        'imputation_method': IMPUTATION_METHOD,
        'models': list(MODELS.keys()),
        'feature_sets': list(dynamic_feature_sets.keys()),
        'data_files': {
            'structured': STRUCTURED_FILE_PATH,
            'snow': SNOW_FEATURES_PATH if os.path.exists(SNOW_FEATURES_PATH) else None,
            'embeddings': EMBEDDING_FILES,
            'additional': ADDITIONAL_FEATURES
        }
    }

    with open(os.path.join(run_results_dir, 'config.json'), 'w') as f:
        json.dump(config_summary, f, indent=2)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {run_results_dir}")
    print(f"\nModels evaluated: {', '.join(MODELS.keys())}")
    print(f"Feature sets evaluated: {len(FEATURE_SETS)}")
    print(f"Iterations per evaluation: {N_ITERATIONS}")
    print("\nGenerated files per model:")
    print("  - <model>_summary.csv: Mean ± std for each feature set")
    print("  - <model>_detailed.csv: All scores per iteration")
    print("  - <model>_auc_boxplot.png: AUC comparison box plot")
    print("  - <model>_auc_barplot.png: AUC comparison bar plot")
    print("  - <model>_accuracy_boxplot.png: Accuracy comparison box plot")
    print("  - <model>_accuracy_barplot.png: Accuracy comparison bar plot")
    print("="*80)


if __name__ == "__main__":
    main()
