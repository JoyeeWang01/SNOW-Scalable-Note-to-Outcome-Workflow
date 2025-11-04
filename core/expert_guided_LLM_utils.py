"""
Expert-guided feature extraction functions.

Extracts predefined prostate cancer features using LLM with structured prompts.
"""

from __future__ import annotations

import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from config.expert_guided_LLM_extraction_config import (
    PROSTATE_REGIONS,
    REGIONAL_FEATURE_TYPES,
    ADDITIONAL_CLINICAL_FEATURES,
    TNM_FEATURE,
    DEFAULT_PERCENT_INVOLVED,
    GLEASON_GRADE_GROUPS,
    GLEASON_45_PRIMARY_WEIGHT,
    GLEASON_45_SECONDARY_WEIGHT
)


def create_regional_features_prompt(clinical_note: str) -> str:
    """
    Create prompt for extracting regional prostate features.

    Args:
        clinical_note: Raw clinical pathology report text

    Returns:
        Formatted prompt string
    """
    regions_list = ", ".join(PROSTATE_REGIONS)

    prompt = f"""You are a medical data extraction assistant. Extract prostate cancer features from the following pathology report.

Extract features for these 12 regions: {regions_list}

For each region, extract:
1. percent_involved: Integer percentage of cancer involvement (0-100). If not mentioned, use 0. If given as size ratio (e.g., "0.3 of 1.1 cm"), calculate percentage as (numerator/denominator)*100.
2. gleason_pattern: Gleason grading (e.g., "3+4", "4+5"). Use null if not mentioned.
3. histology: Tissue type (e.g., "adenocarcinoma", "benign", "high-grade PIN"). Use null if not mentioned.

Also extract:
- TNM_N: Lymph node involvement (0 for N0, 1 for N1+, null if not mentioned)

Return ONLY a JSON object with this exact structure (no explanatory text):
{{
  "left_apex_lateral_percent_involved": <integer or 0>,
  "left_apex_lateral_gleason_pattern": "<pattern or null>",
  "left_apex_lateral_histology": "<type or null>",
  ... (repeat for all 12 regions),
  "TNM_N": <0, 1, or null>
}}

Clinical Note:
{clinical_note}

Return only the JSON object:"""

    return prompt


def create_additional_features_prompt(clinical_note: str) -> str:
    """
    Create prompt for extracting additional clinical features.

    Args:
        clinical_note: Raw clinical pathology report text

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a medical data extraction assistant. Extract the following clinical features from this pathology report.

Extract these 6 features:

1. PNI (Perineural Invasion): "Present", "Absent", or "Not Mentioned"
   - Search for: perineural invasion, PNI, nerve involvement

2. cribriform (Cribriform Pattern): "Present", "Absent", or "Not Mentioned"
   - Search for: cribriform, cribriform pattern

3. IDC (Intraductal Carcinoma): "Present", "Absent", or "Not Mentioned"
   - Search for: intraductal carcinoma, IDC

4. EPE (Extraprostatic Extension): "Present", "Absent", or "Not Mentioned"
   - Search for: extraprostatic extension, EPE, extracapsular extension

5. SVI (Seminal Vesicle Invasion): "Present", "Absent", or "Not Mentioned"
   - Search for: seminal vesicle invasion, SVI, seminal vesicle involvement

6. surgical_margin: "Positive", "Negative", or "Not Mentioned"
   - If positive, include location (e.g., "Positive at apex")

Return ONLY a JSON object with this exact structure (no explanatory text):
{{
  "PNI": "<Present/Absent/Not Mentioned>",
  "cribriform": "<Present/Absent/Not Mentioned>",
  "IDC": "<Present/Absent/Not Mentioned>",
  "EPE": "<Present/Absent/Not Mentioned>",
  "SVI": "<Present/Absent/Not Mentioned>",
  "surgical_margin": "<Positive/Negative/Not Mentioned or location if positive>"
}}

Clinical Note:
{clinical_note}

Return only the JSON object:"""

    return prompt


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM JSON response, handling common formatting issues.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        # Try direct JSON parsing
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        # Look for content between first { and last }
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return None


def convert_additional_features(features: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert additional clinical features to numeric encoding.

    Args:
        features: Dict with keys: PNI, cribriform, IDC, EPE, SVI, surgical_margin

    Returns:
        Dict with PNI converted to 0/1/null, others unchanged
    """
    result = features.copy()

    # Convert PNI: Present -> 1, Absent -> 0, Not Mentioned -> null
    if 'PNI' in result:
        pni_value = result['PNI']
        if pni_value == 'Present':
            result['PNI'] = 1
        elif pni_value == 'Absent':
            result['PNI'] = 0
        else:  # Not Mentioned
            result['PNI'] = None

    return result


def calculate_gleason_scores(gleason_pattern: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse Gleason pattern into primary and secondary scores.

    Args:
        gleason_pattern: Pattern like "3+4", "4+5", etc.

    Returns:
        Tuple of (primary_score, secondary_score) or (None, None)
    """
    if not gleason_pattern or gleason_pattern == 'null':
        return None, None

    # Match pattern like "3+4"
    match = re.match(r'(\d+)\+(\d+)', str(gleason_pattern))
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def calculate_gleason_grade(gleason_pattern: str, histology: str) -> int:
    """
    Calculate Gleason grade group from pattern.

    Args:
        gleason_pattern: Pattern like "3+4"
        histology: Tissue type

    Returns:
        Grade group (1-5) or 0 if not adenocarcinoma
    """
    # Only calculate for adenocarcinoma
    if not histology or 'adenocarcinoma' not in str(histology).lower():
        return 0

    if not gleason_pattern or gleason_pattern == 'null':
        return 0

    return GLEASON_GRADE_GROUPS.get(str(gleason_pattern), 0)


def calculate_gleason_45_score(primary: Optional[int], secondary: Optional[int]) -> Optional[float]:
    """
    Calculate weighted Gleason 4/5 score.

    Args:
        primary: Primary Gleason score
        secondary: Secondary Gleason score

    Returns:
        Weighted score (75% primary + 25% secondary) or None
    """
    if primary is None or secondary is None:
        return None

    # Only count scores of 4 or 5
    primary_45 = primary if primary >= 4 else 0
    secondary_45 = secondary if secondary >= 4 else 0

    return (GLEASON_45_PRIMARY_WEIGHT * primary_45 +
            GLEASON_45_SECONDARY_WEIGHT * secondary_45)


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features from regional extractions.

    Args:
        df: DataFrame with regional features

    Returns:
        DataFrame with additional derived features
    """
    result_df = df.copy()

    # Calculate per-region derived features
    for region in PROSTATE_REGIONS:
        gleason_col = f"{region}_gleason_pattern"
        histology_col = f"{region}_histology"

        if gleason_col in result_df.columns:
            # Calculate grade, primary, secondary scores
            grades = []
            primaries = []
            secondaries = []
            gleason_45_scores = []

            for idx, row in result_df.iterrows():
                pattern = row.get(gleason_col)
                histology = row.get(histology_col)

                # Grade
                grade = calculate_gleason_grade(pattern, histology)
                grades.append(grade)

                # Primary and secondary
                primary, secondary = calculate_gleason_scores(pattern)
                primaries.append(primary)
                secondaries.append(secondary)

                # Gleason 4/5 score
                g45 = calculate_gleason_45_score(primary, secondary)
                gleason_45_scores.append(g45)

            result_df[f"{region}_gleason_grade"] = grades
            result_df[f"{region}_primary"] = primaries
            result_df[f"{region}_secondary"] = secondaries
            result_df[f"{region}_gleason_pattern_four_five"] = gleason_45_scores

    # Calculate summary features
    # Max gleason grade
    grade_cols = [f"{region}_gleason_grade" for region in PROSTATE_REGIONS]
    result_df['max_gleason_grade'] = result_df[grade_cols].max(axis=1)

    # Max primary and secondary
    primary_cols = [f"{region}_primary" for region in PROSTATE_REGIONS]
    secondary_cols = [f"{region}_secondary" for region in PROSTATE_REGIONS]
    result_df['max_gleason_primary'] = result_df[primary_cols].max(axis=1)
    result_df['max_gleason_secondary'] = result_df[secondary_cols].max(axis=1)

    # Gleason 4/5 scores
    g45_cols = [f"{region}_gleason_pattern_four_five" for region in PROSTATE_REGIONS]
    result_df['gleason_four_five_max'] = result_df[g45_cols].max(axis=1)
    result_df['gleason_four_five'] = result_df[g45_cols].mean(axis=1)

    # Percent involved statistics
    percent_cols = [f"{region}_percent_involved" for region in PROSTATE_REGIONS]
    result_df['percent_involved_mean'] = result_df[percent_cols].mean(axis=1)
    result_df['percent_involved_max'] = result_df[percent_cols].max(axis=1)

    # Percent positive regions and cores
    # Bilateral regions (6 pairs: apex, mid, base × left/right)
    bilateral_regions = [
        ['left_apex_lateral', 'left_apex_medial', 'right_apex_lateral', 'right_apex_medial'],
        ['left_mid_lateral', 'left_mid_medial', 'right_mid_lateral', 'right_mid_medial'],
        ['left_base_lateral', 'left_base_medial', 'right_base_lateral', 'right_base_medial']
    ]

    def count_positive_bilateral(row):
        positive = 0
        for group in bilateral_regions:
            group_positive = any(row.get(f"{r}_percent_involved", 0) > 0 for r in group)
            if group_positive:
                positive += 1
        return (positive / len(bilateral_regions)) * 100

    result_df['percent_pos_regions'] = result_df.apply(count_positive_bilateral, axis=1)

    # Percent positive cores (out of 12)
    def count_positive_cores(row):
        positive = sum(1 for region in PROSTATE_REGIONS if row.get(f"{region}_percent_involved", 0) > 0)
        return (positive / len(PROSTATE_REGIONS)) * 100

    result_df['percent_pos_cores'] = result_df.apply(count_positive_cores, axis=1)

    # Bilateral (cancer on both left and right)
    def is_bilateral(row):
        left_positive = any(row.get(f"{r}_percent_involved", 0) > 0 for r in PROSTATE_REGIONS if r.startswith('left'))
        right_positive = any(row.get(f"{r}_percent_involved", 0) > 0 for r in PROSTATE_REGIONS if r.startswith('right'))
        return 1 if (left_positive and right_positive) else 0

    result_df['bilateral'] = result_df.apply(is_bilateral, axis=1)

    return result_df
