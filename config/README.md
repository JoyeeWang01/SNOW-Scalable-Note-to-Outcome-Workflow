# Configuration Guide

This guide explains how to customize the SNOW workflow for your specific use case.

## When Do I Need This?

**You DON'T need to configure anything if:**
- ✅ Your data files are named `discharge_notes.csv` and `structured_features.csv`
- ✅ Your CSV has columns `hadm_id` (ID) and `discharge_text` (notes)
- ✅ You're happy with default settings (3 chunks, 10 workers, etc.)

**You NEED to configure if:**
- ⚙️ Your file/column names are different
- ⚙️ You want to adjust processing parameters (speed, parallelism)
- ⚙️ You're running evaluation and need to specify outcome column
- ⚙️ You want to customize prompts or logging

## Table of Contents
- [Overview](#overview)
- [API Configuration](#api-configuration)
- [Workflow Configuration (SNOW_config.py)](#workflow-configuration-snow_configpy)
- [Evaluation Configuration (evaluation_config.py)](#evaluation-configuration-evaluation_configpy)
- [Quick Start Examples](#quick-start-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The SNOW workflow uses three main configuration files:

1. **API Configuration** (`api_*.py`) - LLM provider credentials
2. **Workflow Configuration** (`SNOW_config.py`) - Data paths, processing settings
3. **Evaluation Configuration** (`evaluation_config.py`) - Model evaluation settings

**Most common changes:**
- File paths and column names → Edit `SNOW_config.py`
- API credentials → Edit `api_claude.py` (or your provider)
- Evaluation settings → Edit `evaluation_config.py`

---

## API Configuration

### Purpose
Configure credentials and endpoints for LLM providers (Claude, OpenAI, Gemini).

### Files
- `config/api_claude.py.template` → Copy to `config/api_claude.py`
- `config/api_openai.py.template` → Copy to `config/api_openai.py`
- `config/api_gemini.py.template` → Copy to `config/api_gemini.py`

### Setup Steps

#### 1. Choose Your Provider

In your script (e.g., `scripts/SNOW_feature_definition.py`):
```python
SELECTED_PROVIDER = "claude"  # Options: "claude", "openai", "gemini"
```

#### 2. Copy the Template

```bash
# For Claude
cp config/api_claude.py.template config/api_claude.py

# For OpenAI/Azure OpenAI
cp config/api_openai.py.template config/api_openai.py

# For Gemini
cp config/api_gemini.py.template config/api_gemini.py
```

#### 3. Edit Configuration

**For Claude (Anthropic/AWS Bedrock):**

```python
# config/api_claude.py
url = "YOUR_CLAUDE_API_URL_HERE"
key = "YOUR_API_KEY_HERE"
model = "YOUR_MODEL_ARN_HERE"
llm_provider = "claude"
```

**Example values:**
- AWS Bedrock: `url = "https://bedrock-runtime.us-east-1.amazonaws.com"`
- Model ARN: `model = "arn:aws:bedrock:us-east-1:123456789:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"`
- Azure APIM: `url = "https://your-org.azure-api.net/claude/v1/messages"`

**For OpenAI/Azure OpenAI:**

```python
# config/api_openai.py
key = "YOUR_API_KEY_HERE"
API_VERSION = "2024-08-01-preview"
DEPLOYMENT = "gpt-4"
url = f'YOUR_AZURE_ENDPOINT/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}'
model = DEPLOYMENT
llm_provider = "openai"
```

**Example values:**
- Azure endpoint: `url = "https://your-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview"`
- Deployment: `DEPLOYMENT = "gpt-4"` or `"gpt-35-turbo"`

**For Gemini (Google Vertex AI):**

```python
# config/api_gemini.py
PROJECT_ID = "YOUR_GCP_PROJECT_ID_HERE"
LOCATION = "us-central1"
model = "gemini-1.5-pro-002"
llm_provider = "gemini"
```

**Additional setup for Gemini:**
```bash
# Authenticate with Google Cloud
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Security Notes

- ⚠️ **NEVER commit `api_*.py` files to version control** (they're in `.gitignore`)
- ✅ **Only commit `.template` files** (with placeholder values)
- 🔒 **Keep API keys secure** - rotate if exposed
- 💡 **Use environment variables** for production deployments

---

## Workflow Configuration (SNOW_config.py)

### Purpose
Configure data paths, processing parameters, and logging settings for the SNOW workflow.

### File Location
`config/SNOW_config.py`

### Key Parameters

#### Data Paths

```python
# Directory paths
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CONFIG_DIR)
_DATA_DIR = os.path.join(_ROOT_DIR, 'data')

# Input data files
NOTES_FILE_PATH = os.path.join(_DATA_DIR, 'discharge_notes.csv')
STRUCTURED_FEATURES_FILE_PATH = os.path.join(_DATA_DIR, 'structured_features.csv')

# Column names
NOTES_COL = 'discharge_text'  # Name of column containing clinical notes
ID_COL = 'hadm_id'             # Name of patient ID column
OUTCOME_COL = 'death_30_days'  # Name of outcome column (for evaluation)
```

**How to customize:**
1. **Change input files**: Modify `NOTES_FILE_PATH` to point to your data
2. **Change column names**: Update `NOTES_COL`, `ID_COL`, `OUTCOME_COL` to match your CSV columns
3. **Use different paths**: Change `_DATA_DIR` if your data is elsewhere

**Example - Using different file names:**
```python
NOTES_FILE_PATH = os.path.join(_DATA_DIR, 'pathology_reports.csv')
STRUCTURED_FEATURES_FILE_PATH = os.path.join(_DATA_DIR, 'baseline_features.csv')
NOTES_COL = 'report_text'
ID_COL = 'mrn'
OUTCOME_COL = 'recurrence'
```

#### Feature Proposal & Alignment

```python
# Feature proposal settings
NUM_CHUNKS = 3          # Number of chunks to divide notes into for feature alignment
BATCH_SIZE = 10         # Number of notes per batch in feature alignment
```

**Recommendations:**
- **Small dataset (<300 notes)**: `NUM_CHUNKS = 3`, `BATCH_SIZE = 10`
- **Medium dataset (300-1000 notes)**: `NUM_CHUNKS = 5`, `BATCH_SIZE = 10`
- **Large dataset (>1000 notes)**: `NUM_CHUNKS = 10`, `BATCH_SIZE = 20`

**What these control:**
- `NUM_CHUNKS`: More chunks = faster and more parallel processing
- `BATCH_SIZE`: More notes per batch = better feature coverage but higher LLM cost

#### Feature Extraction & Validation

```python
# Parallel processing
MAX_WORKERS = 10        # Number of parallel workers for extraction/validation
MAX_RETRIES = 3         # Maximum validation retry attempts per feature

# Checkpoint settings
CHECKPOINT_DIR = os.path.join(_ROOT_DIR, 'checkpoints')
SAVE_FREQUENCY = 5      # Save checkpoint every N features
```

**Recommendations:**
- **Local testing**: `MAX_WORKERS = 5` (to avoid rate limits)
- **Production**: `MAX_WORKERS = 10-20` (based on API quota)
- **API rate limits**: Reduce `MAX_WORKERS` if you hit rate limits

**What these control:**
- `MAX_WORKERS`: More workers = faster extraction but more API calls/second
- `MAX_RETRIES`: How many times to re-extract if validation fails
- `SAVE_FREQUENCY`: How often to save progress (lower = more frequent saves)

#### Logging

```python
# Logging settings
LOG_DIR = os.path.join(_ROOT_DIR, 'logs')
DETAILED_LOGGING = False  # Set to True to log all LLM prompts and responses
DETAILED_LOG_DIR = os.path.join(LOG_DIR, 'detailed')
```

**Enable detailed logging for debugging:**
```python
DETAILED_LOGGING = True  # Saves all prompts and responses to files
```

**What gets logged when `DETAILED_LOGGING = True`:**
- Prompt templates (once at startup)
- Every LLM request and response
- Tool calls (if using tool-calling features)
- Alignment decisions for each chunk

**Log files created:**
- `logs/feature_definition_log_{timestamp}.txt` - Main execution log
- `logs/detailed/prompt_templates_feature_definition.txt` - Prompt templates
- `logs/detailed/align_response_chunk0.txt` - Feature alignment responses
- `logs/detailed/extract_response_chunk0_{timestamp}.txt` - Extraction responses
- `logs/detailed/validate_response_{feature_name}_{timestamp}.txt` - Validation responses

#### Text Descriptions

```python
# Dataset descriptions (used in prompts)
NOTES_DESCRIPTION = "discharge summaries from heart failure patients"
OUTCOME_DESCRIPTION = "30-day mortality after hospital discharge"
STRUCTURED_FEATURES_DESCRIPTION = "age, gender, vitals, lab values, and ICD-9 comorbidities"
```

**Customize for your use case:**

**Example - Oncology:**
```python
NOTES_DESCRIPTION = "pathology reports from prostate cancer patients"
OUTCOME_DESCRIPTION = "biochemical failure within 5 years"
STRUCTURED_FEATURES_DESCRIPTION = "age, PSA level, Gleason score, and clinical stage"
```

---

## Evaluation Configuration (evaluation_config.py)

### Purpose
Configure model evaluation settings, feature sets for comparison, and cross-validation parameters.

### File Location
`config/evaluation_config.py`

### Key Parameters

#### Data Paths and Labels

```python
# Root dirs
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CONFIG_DIR)
_DATA_DIR = os.path.join(_ROOT_DIR, 'data/examples')

# Structured features + labels
STRUCTURED_FILE_PATH = os.path.join(_DATA_DIR, 'structured_features.csv')
LABEL_COL = 'death_30_days'

# SNOW features (optional)
SNOW_FEATURES_PATH = os.path.join(_DATA_DIR, 'generated_features.csv')

# Index/notes columns come from SNOW_config.py
from config.SNOW_config import NOTES_FILE_PATH, INDEX_COL, NOTES_COL
```

**How to customize:**
- Point `STRUCTURED_FILE_PATH` and `SNOW_FEATURES_PATH` to your files.
- Set `LABEL_COL` to your outcome column.
- Update `INDEX_COL`/`NOTES_COL` in `SNOW_config.py` if needed.

#### Embeddings and Additional Feature Sets

```python
_EMBEDDINGS_DIR = os.path.join(_ROOT_DIR, 'data/embeddings')

EMBEDDING_FILES = {
    'bow_classic': os.path.join(_EMBEDDINGS_DIR, 'bow_classic.csv'),
    'bow_tfidf': os.path.join(_EMBEDDINGS_DIR, 'bow_tfidf.csv'),
    'gemini': os.path.join(_EMBEDDINGS_DIR, 'gemini_embeddings.csv'),
}

ADDITIONAL_FEATURES = {
    # 'custom_features': 'path/to/features.csv',
}
```

**Notes:**
- Keys in `EMBEDDING_FILES` must match names used in `FEATURE_SETS`.
- Only files that exist are loaded; missing files are skipped.

#### Feature Sets (sources-based)

```python
FEATURE_SETS = {
    'Baseline': {
        'description': 'Baseline features only',
        'sources': ['structured']
    },
    'Baseline + SNOW': {
        'description': 'Baseline + SNOW-extracted features',
        'sources': ['structured', 'snow']
    },
    'Baseline + BoW Classic': {
        'description': 'Baseline + BoW classic embeddings',
        'sources': ['structured', 'bow_classic']
    },
    'Baseline + Gemini Embeddings': {
        'description': 'Baseline + Gemini embeddings',
        'sources': ['structured', 'gemini']
    },
    'Baseline + RFG': {
        'description': 'Baseline + Representative Feature Generation',
        'sources': ['structured', 'rfg']  # Special marker for multi-embedding selection
    }
}
```

**Sources can include:**
- `structured` (baseline features)
- `snow` (SNOW-extracted features)
- embedding keys from `EMBEDDING_FILES`
- keys from `ADDITIONAL_FEATURES`
- `rfg` (special RFG model-selection pipeline)

#### Cross-Validation Settings

```python
N_ITERATIONS = 50
MASTER_SEED = 2048
N_OUTER_FOLDS = 2
N_INNER_FOLDS = 2
INNER_CV_REPEATS = 3

# NLP embedding dimensionality variants
# Use "ALL" or -1 to include the original embedding
TARGET_DIMS = [5, "ALL"]
```

**Recommendations:**
- **Quick testing**: fewer folds + iterations (e.g., `N_ITERATIONS = 5`)
- **Thorough evaluation**: more folds + iterations (e.g., `N_ITERATIONS = 50`)
- For tiny datasets, include `"ALL"` in `TARGET_DIMS` to avoid zero variants.

**What these control:**
- `N_OUTER_FOLDS`: Generalization estimate
- `N_INNER_FOLDS` + `INNER_CV_REPEATS`: Hyperparameter tuning stability
- `N_ITERATIONS`: Repeat the full nested CV with different seeds

#### Imputation Settings

```python
IMPUTATION_METHOD = 'median'  # 'mean', 'mice', 'svd', 'knn', 'median'

# MICE
MICE_MAX_ITER = 5
MICE_ESTIMATOR = 'linear_regression'
MICE_N_NEAREST_FEATURES = 20
MICE_SKIP_COMPLETE = True

# SVD
SVD_N_COMPONENTS = 3
SVD_MAX_ITER = 500
SVD_TOL = 1e-4

# KNN
KNN_N_NEIGHBORS = 5
```

#### Model Settings

```python
MODELS = {
    'LogisticRegression': {
        'class': 'sklearn.linear_model.LogisticRegression',
        'params': {
            'random_state': None,
            'max_iter': 3000
        },
        'param_grid': {
            'C': np.logspace(-5, 1, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'KNN': {
        'class': 'sklearn.neighbors.KNeighborsClassifier',
        'params': {},
        'param_grid': {
            'n_neighbors': [5, 10, 20, 30, 40, 50],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
}
```

#### Model Selection (Optional)

```python
MODELS_FOR_SELECTION = MODELS
RUN_MODEL_SELECTION = False
```

When `RUN_MODEL_SELECTION=True`, the inner CV chooses the best model type
and hyperparameters across `MODELS_FOR_SELECTION`.

#### Metrics and Scoring

```python
INNER_CV_SCORING = 'auc'   # 'auc', 'aupr', 'f1', 'accuracy'
METRICS = ['auc', 'aupr', 'f1']
```

#### Output and Visualization

```python
RESULTS_DIR = "evaluation_results"
SAVE_DETAILED_RESULTS = True
SAVE_FEATURE_IMPORTANCE = True

PLOT_FIGSIZE = (14, 8)
PLOT_DPI = 300
TOP_N_FEATURES = 20
BOX_PLOT_COLORS = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightpurple']
```

**Typical outputs per run:**
- `evaluation_results/evaluation_<timestamp>/config.json` (config snapshot)
- `<model>_summary.csv` and `<model>_detailed.csv`
- `<model>_<feature_set>_detailed_results.pkl` (includes predictions)
- `roc_curves/` with ROC plots + raw ROC data

#### Permutation Tests and Logging

```python
ENABLE_PERMUTATION_TESTS = False
PERMUTATION_TEST_PAIRS = [
    ('Baseline', 'Baseline + SNOW'),
]
PERMUTATION_TEST_MODELS = ['LogisticRegression']
PERMUTATION_N_PERMUTATIONS = 10000
PERMUTATION_SEED = 42

VERBOSITY = 1  # 0=silent, 1=progress, 2=detailed
```

---

## Quick Start Examples

### Example 1: Testing with Synthetic Data

```python
# config/SNOW_config.py
NOTES_FILE_PATH = os.path.join(_DATA_DIR, 'examples', 'discharge_notes.csv')
STRUCTURED_FEATURES_FILE_PATH = os.path.join(_DATA_DIR, 'examples', 'structured_features.csv')

NOTES_COL = 'discharge_text'
ID_COL = 'hadm_id'

NUM_CHUNKS = 1          # Just one chunk for testing
BATCH_SIZE = 3          # Small batch size
MAX_WORKERS = 2         # Limited parallelism
DETAILED_LOGGING = True # Enable debugging
```

---

## Troubleshooting

### API Configuration Issues

**Problem**: `ModuleNotFoundError: No module named 'config.api_claude'`

**Solution**: Copy the template file
```bash
cp config/api_claude.py.template config/api_claude.py
# Then edit config/api_claude.py with your credentials
```

**Problem**: API authentication errors

**Solutions**:
- Check API key is correct (no spaces or line breaks)
- Verify endpoint URL is accessible
- For Azure: Confirm deployment name matches
- For Gemini: Run `gcloud auth application-default login`
- Check API quotas and billing

### Workflow Configuration Issues

**Problem**: `FileNotFoundError: data/discharge_notes.csv not found`

**Solution**: Update path in `SNOW_config.py`
```python
NOTES_FILE_PATH = os.path.join(_DATA_DIR, 'your_actual_filename.csv')
```

**Problem**: `KeyError: 'discharge_text'`

**Solution**: Update column name in `SNOW_config.py`
```python
NOTES_COL = 'your_actual_column_name'
```

**Problem**: Rate limit errors from API

**Solution**: Reduce parallel workers
```python
MAX_WORKERS = 5  # Reduce from 10 to 5
```

### Evaluation Configuration Issues

**Problem**: `FileNotFoundError: data/embeddings/bow_classic.csv not found`

**Solution**: Either generate embeddings or remove from evaluation
```python
# Option 1: Generate embeddings
python generate_embeddings/generate_bow_classic.py

# Option 2: Remove from FEATURE_SETS
FEATURE_SETS = {
    'structured_only': {...},
    'llm_extracted': {...},
    'combined': {...}
    # Remove embedding-based feature sets
}
```

**Problem**: Evaluation runs too slowly

**Solution**: Reduce CV folds and iterations
```python
N_OUTER_FOLDS = 3       # Reduce from 5
N_INNER_FOLDS = 2       # Reduce from 3
N_ITERATIONS = 10       # Reduce from 20
```

**Problem**: Out of memory during evaluation

**Solution**:
1. Use fewer models:
```python
MODELS = {
    'logistic_regression': LogisticRegression(...)  # Remove other models
}
```

2. Process feature sets separately:
```python
# Run evaluation multiple times with different feature sets
FEATURE_SETS = {
    'llm_extracted': {...}  # Run once for this
}
# Then change to:
FEATURE_SETS = {
    'structured_only': {...}  # Run again for this
}
```

---

## Additional Resources

- **Data Setup**: See [DATA_SETUP.md](../DATA_SETUP.md) for data file requirements
- **README**: See [README.md](../README.md) for overall project documentation
- **Code Organization**: See [CODE_ORGANIZATION.md](../docs/CODE_ORGANIZATION.md) for architecture details
- **Logging Guide**: See [LOGGING_GUIDE.md](../docs/LOGGING_GUIDE.md) for detailed logging configuration
