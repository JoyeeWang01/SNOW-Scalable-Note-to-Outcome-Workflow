# SNOW: Scaling Clinician-Grade Feature Generation from Clinical Notes with Multi-Agent Language Models

A modular Python framework for proposing, extracting, validating, and aggregating clinician-grade features from unstructured clinical text using multiple LLM providers (Gemini, Claude, OpenAI).

Applicable to various clinical domains including cardiology, oncology, and other medical specialties.

📄 **Paper:** [Scaling Clinician-Grade Feature Generation from Clinical Notes with Multi-Agent Language Models](https://arxiv.org/abs/2508.01956)
👥 **Authors:** Jiayi (Joyee) Wang, Jacqueline Jil Vallon, Nikhil V. Kotha, Neil Panjwani, Xi Ling, Margaret Redfield, Sushmita Vij, Sandy Srinivas, John Leppert, Mark K. Buyyounouski, Mohsen Bayati
📅 **arXiv:** 2508.01956

## Overview

This project implements **SNOW (Scalable Note-to-Outcome Workflow)**, a transparent multi-agent large language model (LLM) system described in "[Scaling Clinician-Grade Feature Generation from Clinical Notes with Multi-Agent Language Models](https://arxiv.org/abs/2508.01956)" (Wang et al., 2025). SNOW is designed to scale a rigorous clinician feature generation (CFG) workflow with optional human oversight, producing interpretable structured features from unstructured clinical notes.

The framework is **domain-agnostic** and has been validated on:
- **Prostate cancer** 5-year recurrence prediction from clinical notes (Stanford cohort)
- **Heart failure with preserved ejection fraction (HFpEF)** mortality prediction from discharge summaries (MIMIC-IV cohort)
- Other clinical prediction tasks

### Results (from the paper)

- **Prostate cancer recurrence (N=147):** SNOW (AUC-ROC `0.767 ± 0.041`) matches manual CFG (`0.762 ± 0.026`) and outperforms structured baselines, clinician-guided LLM extraction, and multiple representational feature generation (RFG) baselines.
- **Efficiency:** once configured, SNOW generates a full patient-level feature table in ~12 hours with ~5 hours clinician oversight (≈48× reduction in expert effort vs manual CFG).
- **HFpEF external validation (N=2,084):** SNOW outperforms baseline and RFG methods for **30-day mortality** (AUC-ROC `0.851 ± 0.008`) and **1-year mortality** (AUC-ROC `0.763 ± 0.003`), without task-specific tuning.

### Workflow Architecture

The framework provides a 3-step automated workflow:

1. **Feature Definition** - Proposes relevant clinical features from notes, then aligns the feature list by reviewing actual notes to update, refine, or remove features based on what's available in the dataset
2. **Feature Extraction & Validation** - Extracts feature values from all patient notes and validates extracted values for accuracy and consistency
3. **Feature Aggregation** - Aggregates multi-valued features (e.g., max, mean) into single values per patient for downstream modeling

**Optional:** Model evaluation script is available to compare LLM features against NLP baselines.

## Project Structure

```
.
├── config/              # Configuration files
│   ├── api_claude.py.template      # Claude API template (copy to api_claude.py)
│   ├── api_openai.py.template      # OpenAI API template (copy to api_openai.py)
│   ├── api_gemini.py.template      # Gemini API template (copy to api_gemini.py)
│   ├── api_claude.py               # Your credentials (NOT in repo, gitignored)
│   ├── api_openai.py               # Your credentials (NOT in repo, gitignored)
│   ├── api_gemini.py               # Your credentials (NOT in repo, gitignored)
│   ├── SNOW_config.py              # Data paths and processing configuration
│   ├── evaluation_config.py        # Evaluation configuration
│   ├── paper_configs/              # Configs used for paper experiments
│   └── prompts.py                  # LLM prompt templates
│
├── core/                # Core modules
│   ├── llm_interface.py            # Unified LLM query interface
│   ├── feature_operations.py       # Feature proposal, extraction, validation
│   ├── checkpoint_manager.py       # Checkpoint and progress tracking
│   ├── data_utils.py               # Data transformation utilities
│   ├── file_io.py                  # File I/O operations
│   ├── evaluation.py               # Model evaluation logic
│   ├── ml_models.py                # Machine learning models
│   └── ...                         # Additional utilities
│
├── scripts/             # Individual workflow step scripts
│   ├── SNOW_feature_definition.py       # Step 1: Feature proposal & alignment
│   ├── SNOW_extract_validate_loop.py    # Step 2: Feature extraction & validation
│   ├── SNOW_feature_aggregation.py      # Step 3: Feature aggregation
│   ├── SNOW_evaluation.py               # Optional: Model evaluation
│   └── SNOW_expert_guided_LLM_extraction.py  # Alternative: Expert-guided extraction
│
├── generate_embeddings/ # NLP embedding generation scripts
│   ├── generate_bow_classic.py
│   ├── generate_bow_tfidf.py
│   └── generate_gemini_embeddings.py
│
├── data/                # Data directory
│   ├── examples/                   # Synthetic examples (in repo)
│   │   ├── discharge_notes.csv          # Example clinical notes
│   │   └── structured_features.csv      # Example structured features
│   ├── embeddings/                 # Generated embeddings (gitignored)
│   └── *.csv                       # Your actual data (NOT in repo, gitignored)
│
├── docs/                # Documentation
│   ├── CODE_ORGANIZATION.md        # Detailed code organization guide
│   ├── PARALLEL_PROCESSING.md      # Parallel processing guide
│   └── LOGGING_GUIDE.md            # Logging configuration guide
│
├── MIMIC_SQL/           # MIMIC-IV cohort selection queries
│   ├── README.md                       # Cohort reproduction guide
│   ├── 01_cohort_selection.sql         # Step 1: Select HFpEF patients
│   ├── 02_discharge_notes.sql          # Step 2: Add discharge notes
│   └── 03_structured_features.sql      # Step 3: Gather baseline features
│
├── main.py              # **RUN THIS**: Full workflow orchestrator
├── DATA_SETUP.md        # **START HERE**: Data and API setup guide
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Features

- **Multi-Provider Support**: Seamlessly switch between Gemini, Claude, and OpenAI
- **Checkpoint/Resume**: Built-in checkpointing for fault-tolerant long-running processes
- **Parallel Processing**: Concurrent feature extraction and validation with configurable workers
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Tool Calling**: LLM agents can query clinical notes on-demand during feature proposal
- **Iterative Refinement**: Validation loop with automatic re-extraction when needed

## Installation

### Prerequisites

- Python 3.8+
- Access to at least one LLM provider:
  - **Claude**: AWS Bedrock or Anthropic API
  - **OpenAI**: Azure OpenAI or OpenAI API
  - **Gemini**: Google Cloud Vertex AI

### Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Provider-specific (install as needed):
pip install google-cloud-aiplatform  # For Gemini
pip install openai                   # For OpenAI/Azure OpenAI
pip install requests                 # For Claude API calls
```

### First-Time Setup

**IMPORTANT**: Before running any scripts, you must:

1. **Configure API credentials** - See [DATA_SETUP.md](DATA_SETUP.md#api-configuration)
2. **Prepare data files** - See [DATA_SETUP.md](DATA_SETUP.md#data-files-setup)
3. **Verify security** - Ensure no sensitive data is in version control

```bash
# Quick check
git status  # Should NOT show data/*.csv or config/api_*.py (unless .template)
```

## Quick Start

> **⚠️ IMPORTANT**: This repository does NOT include actual patient data or API credentials for security and privacy reasons.

### Prerequisites Checklist

Before running the SNOW workflow, ensure you have:
- [ ] Python 3.8+ installed
- [ ] Access to an LLM API (Claude, OpenAI, or Gemini)
- [ ] Clinical notes data (your own or use provided examples)

### Step-by-Step Setup

#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Configure API Access

Choose ONE LLM provider and set up credentials:

**For Claude:**
```bash
cp config/api_claude.py.template config/api_claude.py
# Edit config/api_claude.py and add your API credentials
```

**For OpenAI/Azure OpenAI:**
```bash
cp config/api_openai.py.template config/api_openai.py
# Edit config/api_openai.py and add your API credentials
```

**For Gemini:**
```bash
cp config/api_gemini.py.template config/api_gemini.py
# Edit config/api_gemini.py and add your GCP project ID
gcloud auth application-default login  # Authenticate with Google Cloud
```

📖 **Detailed API setup**: See [DATA_SETUP.md](DATA_SETUP.md#api-configuration)

#### Step 3: Prepare Data

**Option A: Use Example Data (for testing)**
```bash
cp data/examples/discharge_notes.csv data/discharge_notes.csv
```

**Option B: Use Your Own Data**
```bash
# Place your clinical notes file in data/
# File must have: patient ID column + clinical text column
# Example: data/my_clinical_notes.csv
```

**Option C: Reproduce MIMIC-IV Cohort** *(for researchers with MIMIC access)*
```bash
# Follow instructions in MIMIC_SQL/README.md to:
# 1. Select HFpEF patients from MIMIC-IV
# 2. Add discharge notes
# 3. Gather baseline structured features
# Then export results to data/ directory
```

📖 **Reproduce MIMIC-IV cohort**: See [MIMIC_SQL/README.md](MIMIC_SQL/README.md)

**Important**:
- Your data files are automatically excluded from git (in `.gitignore`)
- Ensure data is de-identified before use
- See `data/examples/README.md` for expected data format

📖 **Detailed data setup**: See [DATA_SETUP.md](DATA_SETUP.md#what-data-do-i-need)

#### Step 4: Configure Data Paths (if needed)

If your data files have different names or column names, update `config/SNOW_config.py`:

```python
# Update file paths
NOTES_FILE_PATH = os.path.join(_DATA_DIR, 'my_notes.csv')  # Your file name

# Update column names to match your CSV
NOTES_COL = 'clinical_text'  # Name of text column
ID_COL = 'patient_id'         # Name of ID column
```

📖 **All configuration options**: See [config/README.md](config/README.md)

### Step 5: Run the Workflow

**Choose your approach:**

#### 🚀 Option A: Run Full Workflow (Recommended)

Perfect for first-time users or when you want everything automated:

```bash
python main.py
```

**What it does:**
1. **Step 1**: Proposes features from your clinical notes and aligns them by reviewing actual data
2. **Step 2**: Extracts feature values from all notes and validates accuracy
3. **Step 3**: Aggregates multi-valued features into single values per patient

**Output:** `extracted_features_MMDD.csv` with aggregated features ready for modeling

---

#### 🔧 Option B: Run Individual Steps (Advanced)

For more control or to resume from a specific step:

```bash
cd scripts
```

**Step 1: Feature Definition**

```bash
cd scripts
python SNOW_feature_definition.py
```

**Output:** `saved_features/unified_features_latest.json`
**What it does:** Proposes features from notes, then aligns them by reviewing actual data

**Step 2: Feature Extraction & Validation**

```bash
python SNOW_extract_validate_loop.py
```

**Output:** `saved_features/extracted_features_df_latest.csv`
**What it does:** Extracts values for each feature from all notes, validates accuracy

**Step 3: Feature Aggregation**

```bash
python SNOW_feature_aggregation.py
```

**Output:** `extracted_features_MMDD.csv`
**What it does:** Aggregates multi-valued features (max, mean, etc.) into single values per patient for downstream modeling

---

#### 📊 Optional: Model Evaluation

**Step 4: Evaluate Models** *(Optional)*

```bash
python SNOW_evaluation.py
```

**Output:** Model performance comparison metrics and visualizations
**What it does:** Compares LLM features vs. NLP baselines using nested cross-validation
**Requires:** Structured features file with outcome column (see [DATA_SETUP.md](DATA_SETUP.md#to-run-snow_evaluationpy-model-evaluation))

---

### What's Next?

After running the 3-step workflow, you can:

1. **Use features for modeling**: Load `extracted_features_MMDD.csv` into your ML pipeline
2. **Evaluate performance** *(optional)*: Run `SNOW_evaluation.py` to compare against baselines (requires outcome variable)
3. **Iterate on features**: Modify prompts in `config/prompts.py` and re-run
4. **Try different domains**: Update dataset and descriptions in `config/SNOW_config.py`

📖 **For configuration details**: See [config/README.md](config/README.md)
📖 **For troubleshooting**: See [DATA_SETUP.md](DATA_SETUP.md#troubleshooting)

---

> **Note**: Script names reference "SNOW" (Scalable Note-to-Outcome Workflow) from the [research paper](https://arxiv.org/abs/2508.01956), but the system is **domain-agnostic** and works for any clinical prediction task.

## Configuration

> 📖 **For detailed configuration instructions, see [config/README.md](config/README.md)**

### Processing Configuration

Edit `config/SNOW_config.py`:

```python
NUM_CHUNKS = 3          # Number of chunks for feature alignment
BATCH_SIZE = 10         # Notes per batch in feature alignment
MAX_WORKERS = 10        # Parallel workers for extraction/validation
```

### Parallel Processing

In `scripts/SNOW_feature_definition.py`, enable parallel chunk processing:

```python
PROCESS_CHUNKS_IN_PARALLEL = True  # Process chunks in parallel (faster)
```

**Benefits:**
- 🚀 Significantly faster feature alignment
- ✅ Processes multiple chunks concurrently
- 📊 Automatic checkpoint support

See `docs/PARALLEL_PROCESSING.md` for detailed usage guide.

### Checkpoint Configuration

Checkpoints are automatically saved to the `checkpoints/` directory. To resume from a checkpoint, simply re-run the script.

### Logging

Each script logs to its own separate file in the `logs/` directory:

- `feature_definition_log_{timestamp}.txt` - Feature proposal and alignment
- `extract_validate_loop_log_{timestamp}.txt` - Feature extraction and validation
- `aggregator_log_{timestamp}.txt` - Feature aggregation

Logs include both console output and are saved with timestamps for debugging.

#### Detailed Logging Mode

Enable detailed logging to capture full LLM prompts and responses in `config/SNOW_config.py`:

```python
DETAILED_LOGGING = True  # Set to True to enable detailed LLM output logging
DETAILED_LOG_DIR = "logs/detailed"  # Directory for detailed logs
```

This logs:
- **Prompt templates**: Saved once at script startup to `prompt_templates_{script_name}.txt`
- **LLM responses**: Separate files for each query
  - `align_response_chunk{X}.txt` - Feature alignment responses (one file per chunk, all batches appended)
  - `extract_response_chunk{X}_{timestamp}.txt` - Feature extraction responses
  - `validate_response_{feature_name}_{timestamp}.txt` - Feature validation responses

See `docs/LOGGING_GUIDE.md` for detailed logging configuration and usage.

## Key Concepts

### Feature Proposal

The system uses an LLM with tool-calling capabilities to:
- Browse clinical notes on-demand
- Identify patterns and features relevant to the prediction target
- Exclude features already available as structured data

### Feature Alignment

Proposed features are aligned with the dataset by:
- Reviewing batches of actual clinical notes
- Marking features as "keep", "drop", or "modify"
- Processing notes in chunks with checkpoint support

### Feature Extraction

For each aligned feature and each patient note:
- LLM extracts the feature value
- Parallel processing with configurable workers
- Progressive submission for efficient concurrency

### Feature Validation

Extracted features are validated by:
- Checking for consistency across the dataset
- Identifying potential extraction errors
- Automatically re-extracting failed features
- Iterating until validation passes or max attempts reached

## Advanced Usage

### Switching Between Providers

Change the `SELECTED_PROVIDER` variable in any script:

```python
SELECTED_PROVIDER = "claude"  # Switch from Gemini to Claude
```

The system automatically loads the appropriate configuration.

### Custom Prompts

Edit prompt templates in `config/prompts.py`:

- `FEATURE_PROPOSAL_TEMPLATE` - Feature proposal prompt
- `FEATURE_EXTRACTION_TEMPLATE` - Feature extraction prompt
- `FEATURE_VALIDATION_TEMPLATE` - Feature validation prompt
- `FEATURE_ALIGNMENT_TEMPLATE` - Feature alignment prompt
- `MERGE_FEATURE_TEMPLATE` - Feature merging prompt
- `AGGREGATION_CODE_TEMPLATE` - Aggregation code generation prompt

### Adding a New Provider

1. Create `config/api_newprovider.py` with required variables (model, url, key, llm_provider)
2. Update `core/llm_interface.py`:
   - Add import in `load_api_config()` function
   - Add provider logic in `query_llm()` and `query_llm_with_tools()` functions

See `docs/CODE_ORGANIZATION.md` for detailed instructions.

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're running scripts from the `scripts/` directory and that `__init__.py` files exist in `config/` and `core/` directories.

### API Errors

Check that:
- API credentials are correctly configured
- You have appropriate permissions for the selected provider
- Rate limits and quotas are not exceeded

### Checkpoint Issues

To clear checkpoints and start fresh:

```bash
rm -rf checkpoints/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2025scaling,
  title={Scaling Clinician-Grade Feature Generation from Clinical Notes with Multi-Agent Language Models},
  author={Wang, Jiayi and Vallon, Jacqueline Jil and Kotha, Nikhil V. and Panjwani, Neil and Ling, Xi and Redfield, Margaret and Vij, Sushmita and Srinivas, Sandy and Leppert, John and Buyyounouski, Mark K. and Bayati, Mohsen},
  journal={arXiv preprint arXiv:2508.01956},
  year={2025},
  doi={10.48550/arXiv.2508.01956},
  url={https://arxiv.org/abs/2508.01956}
}
```

**Paper:** [Scaling Clinician-Grade Feature Generation from Clinical Notes with Multi-Agent Language Models](https://arxiv.org/abs/2508.01956)
