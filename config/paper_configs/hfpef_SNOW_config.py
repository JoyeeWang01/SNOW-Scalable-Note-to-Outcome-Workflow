"""
Data paths and feature configuration for oncology feature extraction.
"""

from pathlib import Path

# ============================================================================
# Data File Paths
# ============================================================================
_CONFIG_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CONFIG_DIR.parent.parent
_DATA_DIR = _ROOT_DIR / "data" / "paper" / "hfpef"

NOTES_FILE_PATH = str(_DATA_DIR / "discharge_texts.csv")

# ============================================================================
# Column Configuration
# ============================================================================
NOTES_COL = 'discharge_text'  # Column containing clinical notes
INDEX_COL = 'hadm_id'

# ============================================================================
# Clinical Context
# ============================================================================
# Description of structured features already available (features NOT to extract from notes)
STRUCTURED_FEATURES_DESCRIPTION = """age at admission, gender, heart rate, systolic blood pressure, oxygen saturation, body temperature, bmi, bicarbonate concentration, creatinine level, hemoglobin concentration, international normalized ratio, platelet count, potassium level, white blood cell count, sodium concentration, peak NT-proBNP, peak troponin-T, and binary indicators for acute_myocardial_infarction, peripheral_vascular_disease, cerebrovascular_disease, dementia, chronic_obstructive_pulmonary_disease, rheumatoid_disease, peptic_ulcer_disease, mild_liver_disease, diabetes, diabetes_complications, hemiplegia_paraplegia, renal_disease, cancer, severe_liver_disease, malignant_cancer, hypertension, coronary artery disease, pulmonary hypertension, atrial fibrillation"""

NOTES_DESCRIPTION = "discharge summaries documenting the reason for admission, hospital course, and discharge plans/instructions. Typical sections include Chief Complaint, History of Present Illness, Past Medical History, Brief Hospital Course, Physical Exam, and Discharge Diagnoses (sections aren’t guaranteed to appear verbatim in every note)"
OUTCOME_DESCRIPTION = "whether the patient died within 1 year of hospital discharge"

# ============================================================================
# Processing Configuration
# ============================================================================
# Master directory for all pipeline runs (used by pipeline orchestrator)
MAIN_RUNS_DIR = "SNOW_MIMIC_1yr_runs"

NUM_CHUNKS = 10  # Number of chunks for parallel processing
BATCH_SIZE = 10  # Number of notes per batch in feature confirmation
MAX_WORKERS = 10  # Number of parallel workers for extraction/validation

# Checkpoint frequency (save progress every N completions)
EXTRACTION_CHECKPOINT_FREQUENCY = 10  # Save extraction progress every N rows
VALIDATION_CHECKPOINT_FREQUENCY = 10  # Save validation progress every N features

# ============================================================================
# Feature Definition Configuration
# ============================================================================
DISCOVER_NEW_FEATURES = False  # Set to True to discover features, False to load existing
PROCESS_CHUNKS_IN_PARALLEL = True  # Set to True to process chunks in parallel, False for sequential
PARALLEL_THREAD_DELAY = 10  # Seconds to wait between starting parallel threads (to avoid API overload)

# ============================================================================
# Detailed Logging Configuration
# ============================================================================
DETAILED_LOGGING = True  # Set to True to enable detailed LLM output logging
DETAILED_LOG_DIR = "logs/detailed"  # Directory for detailed logs
