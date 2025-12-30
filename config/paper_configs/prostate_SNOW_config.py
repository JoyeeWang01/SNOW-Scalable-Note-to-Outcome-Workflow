"""
Data paths and feature configuration for oncology feature extraction.
"""

from pathlib import Path

# ============================================================================
# Data File Paths
# ============================================================================
_CONFIG_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CONFIG_DIR.parent.parent
_DATA_DIR = _ROOT_DIR / "data" / "paper" / "prostate"

NOTES_FILE_PATH = str(_DATA_DIR / "notes_with_mrn.csv")

# ============================================================================
# Column Configuration
# ============================================================================
NOTES_COL = 'clean_text'  # Column containing clinical notes
INDEX_COL = 'MRN'

# ============================================================================
# Clinical Context
# ============================================================================
# Description of structured features already available (features NOT to extract from notes)
STRUCTURED_FEATURES_DESCRIPTION = """age_tx, marital_status, language, race_binary, ethnicity, psa_max, charlson"""

NOTES_DESCRIPTION = "pathology reports of prostate cancer patients" #diagnostic bioposy, prostatectomy reports
OUTCOME_DESCRIPTION = "biological failure following prostate cancer treatment (e.g., radical prostatectomy or radiation therapy)"

# ============================================================================
# Processing Configuration
# ============================================================================
# Master directory for all pipeline runs (used by pipeline orchestrator)
MAIN_RUNS_DIR = "SNOW_prostate_runs"

NUM_CHUNKS = 3  # Number of chunks for parallel processing
BATCH_SIZE = 10  # Number of notes per batch in feature confirmation
MAX_WORKERS = 10  # Number of parallel workers for extraction/validation

# Checkpoint frequency (save progress every N completions)
EXTRACTION_CHECKPOINT_FREQUENCY = 5  # Save extraction progress every N rows
VALIDATION_CHECKPOINT_FREQUENCY = 5  # Save validation progress every N features
PARALLEL_THREAD_DELAY = 10  # Seconds to wait between starting parallel threads (to avoid API overload)

# ============================================================================
# Detailed Logging Configuration
# ============================================================================
DETAILED_LOGGING = True  # Set to True to enable detailed LLM output logging
DETAILED_LOG_DIR = "logs/detailed"  # Directory for detailed logs
