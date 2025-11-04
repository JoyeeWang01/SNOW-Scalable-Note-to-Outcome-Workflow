# Configuration Guide

This guide explains how to customize the SNOW pipeline for your specific use case.

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
- [Pipeline Configuration (SNOW_config.py)](#pipeline-configuration-snow_configpy)
- [Evaluation Configuration (evaluation_config.py)](#evaluation-configuration-evaluation_configpy)
- [Quick Start Examples](#quick-start-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The SNOW pipeline uses three main configuration files:

1. **API Configuration** (`api_*.py`) - LLM provider credentials
2. **Pipeline Configuration** (`SNOW_config.py`) - Data paths, processing settings
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

## Pipeline Configuration (SNOW_config.py)

### Purpose
Configure data paths, processing parameters, and logging settings for the SNOW pipeline.

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

#### Data Paths

```python
# Input data
NOTES_FILE_PATH = os.path.join(_DATA_DIR, 'discharge_notes.csv')
STRUCTURED_FEATURES_FILE = os.path.join(_DATA_DIR, 'structured_features.csv')
LLM_EXTRACTED_FEATURES_FILE = os.path.join(_DATA_DIR, 'extracted_features_latest.csv')

# Embedding files (for NLP baselines)
BOW_CLASSIC_FILE = os.path.join(_DATA_DIR, 'embeddings', 'bow_classic.csv')
BOW_TFIDF_FILE = os.path.join(_DATA_DIR, 'embeddings', 'bow_tfidf.csv')
GEMINI_EMBEDDINGS_FILE = os.path.join(_DATA_DIR, 'embeddings', 'gemini_embeddings.csv')
```

**How to customize:**
- Point to your extracted features CSV
- Point to your embedding files (if using NLP baselines)

#### Feature Sets

```python
# Define which feature sets to evaluate
FEATURE_SETS = {
    'structured_only': {
        'name': 'Structured Features Only',
        'features': STRUCTURED_FEATURE_COLS  # Demographics, vitals, labs
    },
    'llm_extracted': {
        'name': 'LLM-Extracted Features',
        'features': LLM_FEATURE_COLS  # Features extracted by SNOW
    },
    'combined': {
        'name': 'Structured + LLM',
        'features': STRUCTURED_FEATURE_COLS + LLM_FEATURE_COLS
    },
    'bow_classic': {
        'name': 'Bag-of-Words (Count)',
        'embedding_file': BOW_CLASSIC_FILE
    },
    'bow_tfidf': {
        'name': 'Bag-of-Words (TF-IDF)',
        'embedding_file': BOW_TFIDF_FILE
    },
    'gemini_embeddings': {
        'name': 'Gemini Embeddings',
        'embedding_file': GEMINI_EMBEDDINGS_FILE
    }
}
```

**Customize which models to compare:**

**Minimal comparison (LLM vs. Structured):**
```python
FEATURE_SETS = {
    'structured_only': {...},
    'llm_extracted': {...},
    'combined': {...}
}
```

**Full comparison (include NLP baselines):**
```python
FEATURE_SETS = {
    'structured_only': {...},
    'llm_extracted': {...},
    'combined': {...},
    'bow_classic': {...},
    'bow_tfidf': {...},
    'gemini_embeddings': {...}
}
```

#### Cross-Validation Settings

```python
# Nested cross-validation parameters
OUTER_CV_FOLDS = 5      # Outer folds for test set evaluation
INNER_CV_FOLDS = 3      # Inner folds for hyperparameter tuning
N_ITER = 20             # Number of random search iterations
RANDOM_STATE = 42       # Random seed for reproducibility
```

**Recommendations:**
- **Quick testing**: `OUTER_CV_FOLDS = 3`, `INNER_CV_FOLDS = 2`, `N_ITER = 10`
- **Standard evaluation**: `OUTER_CV_FOLDS = 5`, `INNER_CV_FOLDS = 3`, `N_ITER = 20`
- **Thorough evaluation**: `OUTER_CV_FOLDS = 10`, `INNER_CV_FOLDS = 5`, `N_ITER = 50`

**What these control:**
- `OUTER_CV_FOLDS`: More folds = better estimate of generalization but slower
- `INNER_CV_FOLDS`: Used for hyperparameter tuning
- `N_ITER`: More iterations = better hyperparameter search but slower

#### Model Settings

```python
# Machine learning models
MODELS = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'random_forest': RandomForestClassifier(random_state=RANDOM_STATE),
    'xgboost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
}

# Hyperparameter grids
PARAM_GRIDS = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
}
```

**Customize models:**

**Use only logistic regression (fastest):**
```python
MODELS = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
}
PARAM_GRIDS = {
    'logistic_regression': {'C': [0.01, 0.1, 1, 10, 100]}
}
```

**Add neural network:**
```python
from sklearn.neural_network import MLPClassifier

MODELS = {
    'logistic_regression': LogisticRegression(...),
    'mlp': MLPClassifier(random_state=RANDOM_STATE)
}
PARAM_GRIDS = {
    'logistic_regression': {...},
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01]
    }
}
```

#### Output Settings

```python
# Results output
RESULTS_DIR = os.path.join(_ROOT_DIR, 'evaluation_results')
SAVE_MODELS = False     # Whether to save trained models
SAVE_PREDICTIONS = True # Whether to save predictions
```

**What gets saved:**
- `evaluation_results/metrics_summary.csv` - Performance metrics for all models
- `evaluation_results/nested_cv_results.json` - Detailed CV results
- `evaluation_results/feature_importance.csv` - Feature importance (if applicable)
- `evaluation_results/roc_curves.png` - ROC curve comparison plot

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

### Pipeline Configuration Issues

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

**Problem**: `FileNotFoundError: embeddings/bow_classic.csv not found`

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
OUTER_CV_FOLDS = 3      # Reduce from 5
INNER_CV_FOLDS = 2      # Reduce from 3
N_ITER = 10             # Reduce from 20
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

- **Data Setup**: See [DATA_SETUP.md](DATA_SETUP.md) for data file requirements
- **README**: See [README.md](README.md) for overall project documentation
- **Code Organization**: See [docs/CODE_ORGANIZATION.md](docs/CODE_ORGANIZATION.md) for architecture details
- **Logging Guide**: See [docs/LOGGING_GUIDE.md](docs/LOGGING_GUIDE.md) for detailed logging configuration
