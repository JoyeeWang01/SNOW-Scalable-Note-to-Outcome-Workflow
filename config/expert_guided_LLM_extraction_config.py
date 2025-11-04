"""
Configuration for fixed-schema feature extraction from prostate pathology reports.

This defines a predefined set of 37 prostate cancer features to extract,
as a comparison baseline to SNOW's dynamic feature proposal.
"""

# ============================================================================
# Prostate Anatomical Regions (12 regions)
# ============================================================================

PROSTATE_REGIONS = [
    'left_apex_lateral',
    'left_apex_medial',
    'left_mid_lateral',
    'left_mid_medial',
    'left_base_lateral',
    'left_base_medial',
    'right_apex_lateral',
    'right_apex_medial',
    'right_mid_lateral',
    'right_mid_medial',
    'right_base_lateral',
    'right_base_medial',
]

# ============================================================================
# Regional Features (3 per region = 36 features)
# ============================================================================

REGIONAL_FEATURE_TYPES = [
    'percent_involved',  # Integer percentage of cancer involvement
    'gleason_pattern',   # Gleason grading (e.g., "3+4", "4+5")
    'histology',         # Tissue type (e.g., "adenocarcinoma", "benign")
]

# ============================================================================
# Additional Clinical Features (6 features)
# ============================================================================

ADDITIONAL_CLINICAL_FEATURES = {
    'PNI': {
        'name': 'Perineural Invasion',
        'values': ['Present', 'Absent', 'Not Mentioned'],
        'synonyms': ['perineural invasion', 'PNI', 'nerve involvement']
    },
    'cribriform': {
        'name': 'Cribriform Pattern',
        'values': ['Present', 'Absent', 'Not Mentioned'],
        'synonyms': ['cribriform', 'cribriform pattern']
    },
    'IDC': {
        'name': 'Intraductal Carcinoma',
        'values': ['Present', 'Absent', 'Not Mentioned'],
        'synonyms': ['intraductal carcinoma', 'IDC']
    },
    'EPE': {
        'name': 'Extraprostatic Extension',
        'values': ['Present', 'Absent', 'Not Mentioned'],
        'synonyms': ['extraprostatic extension', 'EPE', 'extracapsular extension']
    },
    'SVI': {
        'name': 'Seminal Vesicle Invasion',
        'values': ['Present', 'Absent', 'Not Mentioned'],
        'synonyms': ['seminal vesicle invasion', 'SVI', 'seminal vesicle involvement']
    },
    'surgical_margin': {
        'name': 'Surgical Margin Status',
        'values': ['Positive', 'Negative', 'Not Mentioned'],
        'notes': 'Include location if positive'
    }
}

# ============================================================================
# TNM Staging Feature
# ============================================================================

TNM_FEATURE = 'TNM_N'  # Lymph node involvement (0/1/null)

# ============================================================================
# Extraction Instructions
# ============================================================================

# Default values for missing data
DEFAULT_PERCENT_INVOLVED = 0  # If region not mentioned
DEFAULT_GLEASON_PATTERN = None  # null for missing
DEFAULT_HISTOLOGY = None  # null for missing

# Percentage calculation rules
PERCENTAGE_CALCULATION_RULE = "If size is given as ratio (e.g., '0.3 of 1.1 cm'), calculate percentage as (numerator/denominator)*100"

# Gleason grade group mapping (derived feature)
GLEASON_GRADE_GROUPS = {
    # Gleason pattern -> Grade Group (1-5)
    # Only for adenocarcinoma; others get 0
    '3+3': 1,
    '3+4': 2,
    '4+3': 3,
    '4+4': 4,
    '3+5': 4,
    '5+3': 4,
    '4+5': 5,
    '5+4': 5,
    '5+5': 5,
}

# Gleason 4/5 score weighting
GLEASON_45_PRIMARY_WEIGHT = 0.75
GLEASON_45_SECONDARY_WEIGHT = 0.25

# ============================================================================
# Output Configuration
# ============================================================================

# Output file naming
OUTPUT_PREFIX = "fixed_schema_extraction"

# Derived features to calculate
CALCULATE_DERIVED_FEATURES = True

DERIVED_FEATURES = [
    # Per-region derived features (12 each):
    'gleason_grade',  # Grade group (1-5, 0 if not adenocarcinoma)
    'primary',  # Primary Gleason score
    'secondary',  # Secondary Gleason score
    'gleason_pattern_four_five',  # Weighted Gleason 4/5 score

    # Summary features:
    'max_gleason_grade',  # Max grade across regions
    'max_gleason_primary',  # Max primary score
    'max_gleason_secondary',  # Max secondary score
    'gleason_four_five_max',  # Max Gleason 4/5 score
    'gleason_four_five',  # Mean Gleason 4/5 score
    'percent_involved_mean',  # Mean involvement
    'percent_involved_max',  # Max involvement
    'percent_pos_regions',  # % positive bilateral regions (out of 6)
    'percent_pos_cores',  # % positive cores (out of 12)
    'bilateral',  # Cancer on both sides (0/1)
]
