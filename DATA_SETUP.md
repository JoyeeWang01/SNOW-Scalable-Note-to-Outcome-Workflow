# Data Setup Guide

This guide explains how to set up API credentials and data files to run the SNOW (Scalable-Note-to-Outcome-Workflow) workflow.

## Table of Contents
- [Quick Reference](#quick-reference)
- [What Data Do I Need?](#what-data-do-i-need)
- [API Configuration](#api-configuration)
- [Data Files Setup](#data-files-setup)
- [Directory Structure](#directory-structure)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Reference

**Minimum required to run SNOW:**
1. ✅ One LLM API configured (Claude, OpenAI, or Gemini)
2. ✅ Clinical notes CSV file with ID and text columns
3. ✅ Data is de-identified (no PHI/PII)

**Time estimate:** ~15 minutes for setup

---

## Overview

This repository contains code for multi-agent LLM-based clinical feature generation.

⚠️ **IMPORTANT**: No actual patient data or API credentials are included in this repository for security and privacy reasons.

**Before running the workflow:**
1. Configure API credentials for ONE LLM provider
2. Prepare clinical notes data file
3. Ensure proper de-identification of any patient data

---

## What Data Do I Need?

The data requirements depend on which scripts you want to run:

### To Run `main.py` or `SNOW_feature_definition.py` + `SNOW_extract_validate_loop.py`

**Minimum Required:**
- **Clinical notes file** (e.g., `data/discharge_notes.csv`)
  - Must contain: Patient ID column + Clinical text column
  - Example: `hadm_id`, `discharge_text`

**Optional:**
- **Structured features file** (e.g., `data/structured_features.csv`)
  - Only needed if you want to exclude already-available features from LLM extraction
  - Example: Demographics, vitals, labs that you already have
  - **Format requirement**: Should contain only baseline features + optional index column (see evaluation section below for details)

**What you get:** A CSV file with LLM-generated features for each patient

### To Run `SNOW_evaluation.py` *(Optional - Model Evaluation)*

**This script is optional** and only needed if you want to compare LLM-generated features against baseline methods.

**Required (all of the above plus):**
- **Structured features file** with baseline features AND outcome column

  **⚠️ IMPORTANT - File Format Requirements:**
  - **MUST include**: Exactly 1 outcome column (e.g., `death_30_days`, `readmission`)
  - **MUST include**: Baseline features for comparison (demographics, vitals, labs, etc.)
  - **OPTIONAL**: 1 index/ID column (e.g., `patient_id`, `hadm_id`)
  - **⚠️ ALL other columns will be treated as baseline features** and included in model evaluation
  - Do NOT include any extra metadata, timestamps, or non-feature columns unless you want them used as features

  **Example correct structure:**
  ```
  patient_id, age, gender, bmi, heart_rate, creatinine, death_30_days
  1001,       65,  1,      28.5, 85,         1.2,        0
  1002,       72,  0,      31.2, 92,         1.8,        1
  ```
  ✅ Correct: 1 ID column + 5 baseline features + 1 outcome

  ❌ Incorrect: Including columns like `admission_date`, `hospital_name`, `notes_length` will cause them to be used as features
- **LLM-generated features** (from running `SNOW_extract_validate_loop.py`)

**Optional (for NLP baseline comparison):**
- **Embedding files** in `data/embeddings/`
  - `bow_classic.csv` - Bag-of-words features
  - `bow_tfidf.csv` - TF-IDF features
  - `gemini_embeddings.csv` - Gemini embeddings
  - Generate these using scripts in `generate_embeddings/` folder

**What you get:** Model performance comparison (LLM features vs baselines), nested CV results, visualizations

### Quick Reference Table

| Script | Clinical Notes | Structured Features | Outcome Column | Embeddings |
|--------|---------------|---------------------|----------------|------------|
| `SNOW_feature_definition.py` | ✅ Required | ⚠️ Optional | ❌ Not needed | ❌ Not needed |
| `SNOW_extract_validate_loop.py` | ✅ Required | ⚠️ Optional | ❌ Not needed | ❌ Not needed |
| `SNOW_feature_aggregation.py` | ❌ Not needed | ❌ Not needed | ❌ Not needed | ❌ Not needed |
| `main.py` (full workflow) | ✅ Required | ⚠️ Optional | ❌ Not needed | ❌ Not needed |
| `SNOW_evaluation.py` | ✅ Required | ✅ Required | ✅ Required | ⚠️ Optional |

**Legend:**
- ✅ **Required** - Script will not run without this
- ⚠️ **Optional** - Script will run but with limited functionality
- ❌ **Not needed** - Script does not use this data

---

## API Configuration

**You need access to ONE of these LLM providers:**
- **Claude** (Anthropic via AWS Bedrock or direct API)
- **OpenAI** (Azure OpenAI or standard OpenAI API)
- **Gemini** (Google Cloud Vertex AI)

### Quick Setup Steps

1. **Copy template file** for your chosen provider
2. **Add your credentials** to the file
3. **Done!** The workflow will automatically load your config

### Detailed Configuration

#### Step 1: Choose Your Provider

Pick ONE provider you have access to. You'll set it when running scripts:

```python
# In scripts/SNOW_feature_definition.py (and other scripts)
SELECTED_PROVIDER = "claude"  # Options: "claude", "openai", "gemini"
```

#### Step 2: Configure API Credentials

#### Option A: Claude (Anthropic)

1. Copy the template:
   ```bash
   cp config/api_claude.py.template config/api_claude.py
   ```

2. Edit `config/api_claude.py` and fill in:
   ```python
   url = "YOUR_CLAUDE_API_URL_HERE"
   key = "YOUR_API_KEY_HERE"
   model = "YOUR_MODEL_ARN_HERE"
   ```

3. Obtain credentials from:
   - **AWS Bedrock**: If using Claude via AWS Bedrock, you'll need an ARN like:
     ```
     arn:aws:bedrock:REGION:ACCOUNT_ID:inference-profile/MODEL_ID
     ```
   - **Direct API**: If using Anthropic's direct API, set URL to `https://api.anthropic.com/v1/messages`
   - **Azure API Management**: If using Azure APIM proxy (like Stanford), use your organization's endpoint

#### Option B: OpenAI / Azure OpenAI

1. Copy the template:
   ```bash
   cp config/api_openai.py.template config/api_openai.py
   ```

2. Edit `config/api_openai.py` and fill in:
   ```python
   key = "YOUR_API_KEY_HERE"
   DEPLOYMENT = "gpt-4"  # Your deployment name
   URL = f'YOUR_AZURE_ENDPOINT_HERE/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}'
   ```

3. For Azure OpenAI:
   - Get your endpoint from Azure Portal
   - Get your API key from Azure Portal → Your Resource → Keys and Endpoint
   - Note your deployment name (e.g., "gpt-4", "gpt-35-turbo")

4. For standard OpenAI API:
   - Modify the client initialization to use standard OpenAI instead of AzureOpenAI
   - Get your API key from platform.openai.com

#### Option C: Google Gemini

1. Copy the template:
   ```bash
   cp config/api_gemini.py.template config/api_gemini.py
   ```

2. Edit `config/api_gemini.py` and fill in:
   ```python
   PROJECT_ID = "YOUR_GCP_PROJECT_ID_HERE"
   LOCATION = "us-central1"  # Or your preferred region
   ```

3. Set up Google Cloud authentication:
   ```bash
   # Install Google Cloud SDK
   # https://cloud.google.com/sdk/docs/install

   # Authenticate
   gcloud auth application-default login

   # Set project
   gcloud config set project YOUR_PROJECT_ID
   ```

4. Enable Vertex AI API:
   - Go to Google Cloud Console
   - Enable "Vertex AI API" for your project
   - Ensure billing is enabled

### Step 3: Verify Configuration

The configuration files are automatically loaded by `core/llm_interface.py` based on `SELECTED_PROVIDER`. No additional imports needed in your scripts.

---

## Data Files Setup

### Required Data Files

The workflow expects specific CSV files in the `data/` directory:

#### For MIMIC Heart Failure Workflow:

1. **`data/discharge_notes.csv`** - Clinical notes with outcomes
   - Required columns:
     - `hadm_id`: Hospital admission ID (integer, primary key)
     - `discharge_text`: Full discharge summary text (string)

2. **`data/structured_features.csv`** - Structured features and labels
   - Required columns:
     - `hadm_id`: Hospital admission ID (must match discharge_notes.csv)
     - Demographics: `age`, `gender`
     - Vitals: `heart_rate`, `sbp`, `spo2`, `temperature`, `bmi`
     - Labs: `bicarbonate`, `creatinine`, `hemoglobin`, `inr`, `platelet`, `potassium`, `wbc`, `sodium`, `nt_probnp`, `troponin_t`
     - Comorbidities: Binary indicators (0/1)
     - Outcome: `death_30_days` (0/1)

### Data Preparation Checklist

- [ ] **Column names**: Match expected column names exactly (case-sensitive)

- [ ] **Data types**:
  - IDs should be integers or strings
  - Clinical text should be strings (can contain newlines)
  - Numeric features should be float or int
  - Binary indicators should be 0/1

- [ ] **Missing values**:
  - Decide how to handle missing values
  - Use empty strings for missing text
  - Use NaN or empty cells for missing numeric values

- [ ] **File encoding**: Save as UTF-8 CSV files

### Example Data

See `data/examples/` for synthetic example files showing the expected format:
- `data/examples/discharge_notes.csv` - Example discharge summaries
- `data/examples/structured_features.csv` - Example structured features

**These are synthetic data for testing only - not suitable for actual model training.**

### Embeddings (Optional)

If using NLP baselines for evaluation, prepare embedding files:

```
data/embeddings/
├── bow_classic.csv        # Bag-of-words (count vectorization)
├── bow_tfidf.csv          # TF-IDF weighted bag-of-words
└── gemini_embeddings.csv  # Gemini text embeddings
```

Each embedding file should:
- Have same number of rows as your notes file
- First column should be the ID column (hadm_id or MRN)
- Subsequent columns should be numeric features (embedding dimensions)

Generate embeddings using scripts in `generate_embeddings/`:
```bash
python generate_embeddings/generate_bow_classic.py
python generate_embeddings/generate_bow_tfidf.py
python generate_embeddings/generate_gemini_embeddings.py
```

---

## Directory Structure

After setup, your `data/` directory should look like:

```
data/
├── examples/                          # Synthetic examples (committed to repo)
│   ├── discharge_notes.csv
│   ├── structured_features.csv
│   └── README.md
├── embeddings/                        # Generated embeddings (gitignored - generated by user)
│   ├── bow_classic.csv
│   ├── bow_tfidf.csv
│   └── gemini_embeddings.csv
├── discharge_notes.csv                      # YOUR DATA (NOT committed - in .gitignore)
└── structured_features.csv               # YOUR DATA (NOT committed - in .gitignore)
```

**Important**: All actual patient data files are automatically ignored by `.gitignore` and will NOT be committed to version control.

---

## Security Best Practices

### API Keys

- **NEVER commit API keys** to version control
- Use template files (`.template`) for examples
- Keep actual credential files (`api_*.py`) in `.gitignore`
- Rotate keys if accidentally exposed
- Use environment variables for production deployments
- Consider using secret management services (AWS Secrets Manager, Azure Key Vault, Google Secret Manager)

### Patient Data

- **NEVER commit patient data** to public repositories
- All CSV files with patient data are in `.gitignore`
- Use example/synthetic data for testing and development
- Follow institutional IRB and HIPAA guidelines
- Document your data governance process
- Consider encryption at rest for sensitive files
- Use secure file transfer methods (not email)

### Access Control

- Limit access to data files to authorized personnel only
- Use file system permissions appropriately
- Consider using encrypted volumes for data storage
- Audit data access regularly
- Follow principle of least privilege

---

## Troubleshooting

### API Configuration Issues

**Problem**: `ModuleNotFoundError: No module named 'config.api_claude'`

**Solution**:
```bash
# Make sure you copied the template
cp config/api_claude.py.template config/api_claude.py
# Then edit config/api_claude.py with your credentials
```

**Problem**: API authentication errors

**Solution**:
- Verify your API key is correct (no extra spaces)
- Check API endpoint URL is accessible
- For Azure: Ensure subscription key is valid
- For GCP: Run `gcloud auth application-default login`
- Check your API quota/billing

**Problem**: `SELECTED_PROVIDER` not found

**Solution**:
- Make sure you set `SELECTED_PROVIDER` at the top of your script
- Valid options: "claude", "openai", "gemini" (case-sensitive)

### Data File Issues

**Problem**: `FileNotFoundError: data/discharge_notes.csv not found`

**Solution**:
```bash
# Option 1: Use example data for testing
cp data/examples/discharge_notes.csv data/discharge_notes.csv

# Option 2: Place your actual data file
# Make sure it's in the data/ directory with the correct filename
```

**Problem**: `KeyError: 'discharge_text'` or similar column errors

**Solution**:
- Check your CSV has the expected column names (case-sensitive)
- See `data/examples/` for reference formats
- Use `pandas.read_csv('file.csv').columns` to inspect column names

**Problem**: Path issues when running scripts

**Solution**:
- For scripts in `scripts/`, run from that directory:
  ```bash
  cd scripts
  python SNOW_feature_definition.py
  ```
- The scripts use `sys.path.insert()` to find the parent directory

### Permission Issues

**Problem**: Permission denied when reading data files

**Solution**:
```bash
# Check file permissions
ls -la data/

# Fix permissions if needed
chmod 644 data/discharge_notes.csv
```

### Import Errors

**Problem**: `ImportError: attempted relative import with no known parent package`

**Solution**:
- Run scripts from the `scripts/` directory, not the root
- Each script has `sys.path` manipulation at the top

---

## Getting Help

- **Documentation**: See [README.md](README.md), [CLAUDE.md](CLAUDE.md)
- **Issues**: Check existing issues or open a new one on GitHub
- **Architecture**: See [docs/CODE_ORGANIZATION.md](docs/CODE_ORGANIZATION.md)

---

## Quick Start Example

Here's the complete setup workflow from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API (choose one provider)
cp config/api_claude.py.template config/api_claude.py
nano config/api_claude.py  # Add your API credentials

# 3. Use example data for testing
cp data/examples/discharge_notes.csv data/discharge_notes.csv

# 4. Run the workflow
python main.py
```

**That's it!** Your features will be extracted and saved to `extracted_features_MMDD.csv`.

### What Just Happened?

The workflow:
1. ✅ Proposed features from your clinical notes
2. ✅ Aligned features by reviewing actual notes
3. ✅ Extracted feature values for each patient
4. ✅ Validated the extracted values

**Next Steps:**
- Use `saved_features/extracted_features_df_latest.csv` in your ML models
- *(Optional)* Run `scripts/SNOW_evaluation.py` to compare performance against baselines (requires outcome variable)
- Try with your own data by following [Data Files Setup](#data-files-setup)

---

## Additional Resources

### HIPAA De-identification

- [HIPAA Safe Harbor Method](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
- [NIH De-identification Guidance](https://privacyruleandresearch.nih.gov/pr_08.asp)
- Automated tools: Philter, Microsoft Presidio, Google DLP API

### LLM Provider Documentation

- [Anthropic Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [OpenAI API](https://platform.openai.com/docs/introduction)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Google Vertex AI](https://cloud.google.com/vertex-ai/docs)

### Clinical NLP

- [MIMIC-III Database](https://mimic.mit.edu/)
- [i2b2 Datasets](https://www.i2b2.org/NLP/DataSets/)
- [Clinical NLP Tools](https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers)
