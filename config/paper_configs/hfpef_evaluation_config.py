"""
Configuration for model evaluation and nested cross-validation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Resolve repo-relative paths for publication
_CONFIG_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CONFIG_DIR.parent.parent
_DATA_DIR = _ROOT_DIR / "data" / "paper" / "hfpef"
_EMBEDDINGS_DIR = _DATA_DIR / "embeddings_hfpef"

# ============================================================================
# Data File Paths
# ============================================================================

# SNOW-generated features (output from aggregator)
SNOW_FEATURES_PATH = str(_DATA_DIR / "generated_features.csv")

# Structured features and labels (INDEX_COL and NOTES_COL imported from SNOW_config.py)
from config.SNOW_config import NOTES_FILE_PATH, INDEX_COL, NOTES_COL
STRUCTURED_FILE_PATH = str(_DATA_DIR / "baseline_features_1_year.csv")

# Label column name
LABEL_COL = 'death_1_year'  # Biological failure

# Index column for row alignment validation (None to skip validation)
# INDEX_COL and NOTES_COL imported from data_config.py

# Embedding files
# Dictionary format allows easy addition/removal of embeddings
# Keys must match embedding names used in feature sets: 'bow_classic', 'bow_tfidf', 'gemini'
EMBEDDING_FILES = {
    'bow_classic': str(_EMBEDDINGS_DIR / "bow_classic.csv"),
    'bow_tfidf': str(_EMBEDDINGS_DIR / "bow_tfidf.csv"),
    'gemini': str(_EMBEDDINGS_DIR / "gemini_embeddings.csv"),
    'bow_tfidf_llm_cleaned': str(_EMBEDDINGS_DIR / "bow_tfidf_llm_cleaned.csv"),
    'bow_classic_llm_cleaned': str(_EMBEDDINGS_DIR / "bow_classic_llm_cleaned.csv"),
    'gemini_llm_cleaned': str(_EMBEDDINGS_DIR / "gemini_embeddings_llm_cleaned.csv"),

    # Add more embeddings as needed:
    # 'new_embedding': 'path/to/new_embedding.csv',
}

# ============================================================================
# Additional Feature Sets (Optional)
# ============================================================================

# Add additional feature sets here as needed
# Format: 'feature_name': '/path/to/features.csv'
# When empty, no additional features are used
ADDITIONAL_FEATURES = {
    # Example:
    # 'llm_features': 'data/llm_extracted_features.csv',
    # 'manual_features': 'data/manual_annotations.csv',
    # 'clinical_scores': 'data/clinical_scores.csv',
}

# ============================================================================
# Cross-Validation Configuration
# ============================================================================

# Number of evaluation iterations with different random seeds
N_ITERATIONS = 50

# Master random seed (for generating iteration seeds)
MASTER_SEED = 2048

# Nested CV configuration
N_OUTER_FOLDS = 3  # Outer cross-validation folds
N_INNER_FOLDS = 3  # Inner cross-validation folds (for hyperparameter tuning)
INNER_CV_REPEATS = 1  # Number of repeats for inner CV (RepeatedStratifiedKFold)

# NLP Embedding dimensionality variants to test (for NLP evaluation)
# Creates trimmed versions of embeddings with these target dimensions
TARGET_DIMS = [50, 100, 200, 300]

# ============================================================================
# Imputation Configuration
# ============================================================================

# Imputation method: 'mean', 'median', 'mice', 'svd', or 'knn'
IMPUTATION_METHOD = 'median'

# MICE (IterativeImputer) parameters
# Optimized for speed on small datasets:
# - LinearRegression is 5-15x faster than BayesianRidge
# - max_iter=5 as MICE usually stabilizes in a few iterations
# - n_nearest_features=20 for speed gain and reduced overfitting
# - skip_complete=True to avoid modeling fully observed columns
MICE_MAX_ITER = 5
MICE_ESTIMATOR = 'linear_regression'  # Options: 'linear_regression', 'bayesian_ridge'
MICE_N_NEAREST_FEATURES = 20
MICE_SKIP_COMPLETE = True

# SVD imputation parameters
SVD_N_COMPONENTS = 3
SVD_MAX_ITER = 500
SVD_TOL = 1e-4

# KNN imputation parameters
KNN_N_NEIGHBORS = 5

# ============================================================================
# Model Configuration
# ============================================================================

# Models to evaluate
MODELS = {
    'LogisticRegression': {
        'class': 'sklearn.linear_model.LogisticRegression',
        'params': {
            'random_state': None,  # Will be set per iteration
            'max_iter': 5000
        },
        'param_grid': {
            'C': np.logspace(-5, 1, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'RandomFeature': {
        'class': 'core.ml_models.RandomFeatureModel',  # Custom model defined in core/ml_models.py
        'params': {
            'random_state': None  # Will be set per iteration
        },
        'param_grid': {
            'n_components': [256, 512, 1024, 2048, 4096],  # Properly reduced for small dataset (~2000 samples)
            'alpha': np.logspace(-4, 4, 9)
        }
    },
    'KNN': {
        'class': 'sklearn.neighbors.KNeighborsClassifier',
        'params': {},
        'param_grid': {
            'n_neighbors': [5, 10, 20, 30, 40, 50],  # Appropriate for ~2000 samples
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
}

# ============================================================================
# Model Selection Configuration (for automatic model selection)
# ============================================================================

# Multiple models for automatic model selection
# When RUN_MODEL_SELECTION=True, the evaluation will select the best model
# type AND hyperparameters in the inner CV loop
MODELS_FOR_SELECTION = {
    'LogisticRegression': {
        'class': 'sklearn.linear_model.LogisticRegression',
        'params': {
            'random_state': None,  # Will be set per iteration
            'max_iter': 10000
        },
        'param_grid': {
            'C': np.logspace(-4, 2, 13),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'RandomForest': {
        'class': 'sklearn.ensemble.RandomForestClassifier',
        'params': {
            'random_state': None,  # Will be set per iteration
            'n_jobs': -1  # Use all CPU cores
        },
        'param_grid': {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'GradientBoosting': {
        'class': 'sklearn.ensemble.GradientBoostingClassifier',
        'params': {
            'random_state': None  # Will be set per iteration
        },
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'SVM': {
        'class': 'sklearn.svm.SVC',
        'params': {
            'random_state': None,  # Will be set per iteration
            'probability': True  # Required for predict_proba
        },
        'param_grid': {
            'C': np.logspace(-2, 2, 5),
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
}

# Enable/disable model selection mode
# When True, uses MODELS_FOR_SELECTION and selects best model in inner CV
# When False, uses MODELS and evaluates each model separately
RUN_MODEL_SELECTION = False

# ============================================================================
# Feature Set Configuration
# ============================================================================

# Feature sets to evaluate
FEATURE_SETS = {
    'Baseline': {
        'description': 'Baseline features only (from STRUCTURED_FILE_PATH, excluding LABEL)',
        'sources': ['structured']
    },
    'Baseline + BoW TF-IDF': {
        'description': 'Baseline + BoW TF-IDF embeddings',
        'sources': ['structured', 'bow_tfidf']
    },
    'Baseline + CLFG': {
        'description': 'Baseline + Clinician-Guided LLM Feature Generation',
        'sources': ['structured', 'CLFG']
    },
    'Baseline + SNOW': {
        'description': 'Baseline + SNOW-extracted features',
        'sources': ['structured', 'snow']
    },
    'Baseline + CFG': {
        'description': 'Baseline + Clinician Feature Generation',
        'sources': ['structured', 'CFG']
    },
    'Baseline + Gemini': {
        'description': 'Baseline + Gemini',
        'sources': ['structured', 'gemini']
    },
    'Baseline + BoW Classic': {
        'description': 'Baseline + BoW Classic',
        'sources': ['structured', 'bow_classic']
    },
    'Baseline + BoW TF-IDF (LLM-cleaned)': {
        'description': 'Baseline + BoW TF-IDF (LLM-cleaned)',
        'sources': ['structured', 'bow_tfidf_llm_cleaned']
    },
    'Baseline + BoW Classic (LLM-cleaned)': {
        'description': 'Baseline + BoW Classic (LLM-cleaned)',
        'sources': ['structured', 'bow_classic_llm_cleaned']
    },
    'Baseline + Gemini (LLM-cleaned)': {
        'description': 'Baseline + Gemini (LLM-cleaned)',
        'sources': ['structured', 'gemini_llm_cleaned']
    }
}

# ============================================================================
# Evaluation Metrics
# ============================================================================

# Primary metric for model selection
INNER_CV_SCORING = 'auc'  # options: 'auc', 'aupr', 'f1', 'accuracy' (maps to sklearn scorers)

# Metrics to calculate
METRICS = ['auc', 'aupr', 'f1']
# ============================================================================
# Output Configuration
# ============================================================================

# Results directory
RESULTS_DIR = "evaluation_results"

# Whether to save detailed results per iteration
SAVE_DETAILED_RESULTS = True

# Whether to save feature importance
SAVE_FEATURE_IMPORTANCE = True

# ============================================================================
# Visualization Configuration
# ============================================================================

# Plot configuration
PLOT_FIGSIZE = (14, 8)
PLOT_DPI = 300

# Number of top features to show in importance plots
TOP_N_FEATURES = 20

# Box plot configuration
BOX_PLOT_COLORS = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightpurple']

# ============================================================================
# Permutation Test Configuration
# ============================================================================

# Enable permutation tests for statistical comparison of feature sets
ENABLE_PERMUTATION_TESTS = False

# Pairs of feature sets to compare: [(feature_set_1, feature_set_2), ...]
# Tests H₀: Feature Set 2 does NOT outperform Feature Set 1
# User can specify multiple pairs to test different comparisons
#
# Examples:
#   [('Baseline', 'Baseline+SNOW')]  # Test if SNOW improves over baseline
#   [('Baseline', 'Baseline+BoW'), ('Baseline', 'Baseline+TF-IDF')]  # Compare NLP embeddings
#   [('Baseline+SNOW', 'Baseline+SNOW+BoW')]  # Test incremental improvement
#
PERMUTATION_TEST_PAIRS = [
    ('Baseline', 'Baseline + SNOW'),
    ('Baseline + BoW TF-IDF', 'Baseline + SNOW'),
    ('Baseline + SNOW', 'Baseline + BoW TF-IDF')
    # Add more pairs as needed:
    # ('Baseline', 'Baseline+BoW'),
    # ('Baseline+SNOW', 'Baseline+SNOW+BoW'),
]

# Models to run permutation tests for
# Set to None to run for all models, or specify a list of model names
# Examples:
#   None  # Run for all models
#   ['LogisticRegression']  # Only run for LogisticRegression
#   ['LogisticRegression', 'RandomFeature']  # Run for both models
PERMUTATION_TEST_MODELS = ['LogisticRegression']  # Run for all models by default

# Number of permutation iterations (B)
# Higher values give more precise p-values but take longer
# Recommended: 10,000 for final analysis, 1,000 for quick testing
PERMUTATION_N_PERMUTATIONS = 10000

# Random seed for permutation test (ensures reproducibility)
PERMUTATION_SEED = 42

# ============================================================================
# Logging Configuration
# ============================================================================

# Verbosity level (0=silent, 1=progress, 2=detailed)
VERBOSITY = 1
