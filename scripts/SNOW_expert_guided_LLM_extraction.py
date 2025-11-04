"""
Expert-guided feature extraction script.

Extracts predefined prostate cancer features (37 base + 30 derived) as a
comparison baseline to SNOW's dynamic feature proposal.

Usage:
    cd scripts
    python SNOW_expert_guided_LLM_extraction.py
"""

from __future__ import annotations

import os
import json
import pandas as pd
from typing import Dict, Any

# Import modular components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.log_utils import setup_logging, print
setup_logging("expert_guided_extraction")

# Log prompt templates if detailed logging is enabled
from config.SNOW_config import DETAILED_LOGGING, DETAILED_LOG_DIR
if DETAILED_LOGGING:
    os.makedirs(DETAILED_LOG_DIR, exist_ok=True)
    from core.expert_guided_LLM_utils import (
        create_regional_features_prompt,
        create_additional_features_prompt
    )

    # Log prompt templates once
    with open(os.path.join(DETAILED_LOG_DIR, "prompt_templates_fixed_schema.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FIXED SCHEMA EXTRACTION PROMPT TEMPLATES\n")
        f.write("=" * 80 + "\n\n")

        f.write("REGIONAL FEATURES PROMPT TEMPLATE:\n")
        f.write("-" * 80 + "\n")
        f.write(create_regional_features_prompt("<CLINICAL_NOTE>") + "\n\n")

        f.write("ADDITIONAL FEATURES PROMPT TEMPLATE:\n")
        f.write("-" * 80 + "\n")
        f.write(create_additional_features_prompt("<CLINICAL_NOTE>") + "\n\n")

from core.llm_interface import load_api_config
from config.SNOW_config import NOTES_FILE_PATH, NOTES_COL, INDEX_COL
from config.expert_guided_LLM_extraction_config import PROSTATE_REGIONS
from core.llm_interface import query_llm
from core.expert_guided_LLM_utils import (
    create_regional_features_prompt,
    create_additional_features_prompt,
    parse_json_response,
    convert_additional_features,
    calculate_derived_features
)

# Configure LLM provider
SELECTED_PROVIDER = "gemini"  # Options: "gemini", "claude", "openai"


def extract_regional_features(clinical_note: str, llm_provider: str, model: str) -> Dict[str, Any]:
    """
    Extract regional features (36 features) from a clinical note.

    Args:
        clinical_note: Raw pathology report text
        llm_provider: LLM provider name
        model: Model identifier

    Returns:
        Dictionary with regional feature values
    """
    prompt = create_regional_features_prompt(clinical_note)

    try:
        response = query_llm(
            prompt=prompt,
            llm_provider=llm_provider,
            model=model
        )

        # Parse JSON response
        features = parse_json_response(response)

        if features is None:
            print(f"  WARNING: Failed to parse regional features JSON")
            return {}

        return features

    except Exception as e:
        print(f"  ERROR extracting regional features: {e}")
        return {}


def extract_additional_features(clinical_note: str, llm_provider: str, model: str) -> Dict[str, Any]:
    """
    Extract additional clinical features (6 features) from a clinical note.

    Args:
        clinical_note: Raw pathology report text
        llm_provider: LLM provider name
        model: Model identifier

    Returns:
        Dictionary with additional feature values
    """
    prompt = create_additional_features_prompt(clinical_note)

    try:
        response = query_llm(
            prompt=prompt,
            llm_provider=llm_provider,
            model=model
        )

        # Parse JSON response
        features = parse_json_response(response)

        if features is None:
            print(f"  WARNING: Failed to parse additional features JSON")
            return {}

        # Convert PNI to numeric encoding
        features = convert_additional_features(features)

        return features

    except Exception as e:
        print(f"  ERROR extracting additional features: {e}")
        return {}


def extract_all_features(
    df: pd.DataFrame,
    notes_col: str,
    llm_provider: str,
    model: str
) -> pd.DataFrame:
    """
    Extract all features from all notes in the DataFrame.

    Args:
        df: DataFrame with clinical notes
        notes_col: Name of column containing notes
        llm_provider: LLM provider name
        model: Model identifier

    Returns:
        DataFrame with extracted features (37 base features)
    """
    print(f"\nExtracting features from {len(df)} notes...")
    print(f"Using LLM: {llm_provider} / {model}")

    all_features = []

    for idx, row in df.iterrows():
        note = row[notes_col]
        print(f"\nProcessing note {idx + 1}/{len(df)}...")

        # Extract regional features (36 features)
        regional = extract_regional_features(note, llm_provider, model)

        # Extract additional features (6 features)
        additional = extract_additional_features(note, llm_provider, model)

        # Combine all features
        features = {**regional, **additional}
        all_features.append(features)

        print(f"  Extracted {len(features)} features")

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    print(f"\nExtraction complete!")
    print(f"Base features extracted: {features_df.shape[1]}")

    return features_df


def main():
    """Main execution function."""
    print("=" * 80)
    print("FIXED-SCHEMA FEATURE EXTRACTION")
    print("=" * 80)
    print("\nThis script extracts predefined prostate cancer features as a")
    print("comparison baseline to SNOW's dynamic feature proposal.")
    print()
    print(f"Provider: {SELECTED_PROVIDER}")

    # Load API configuration
    api_config = load_api_config(SELECTED_PROVIDER)
    llm_provider = api_config['llm_provider']
    model = api_config['model']

    print(f"LLM Provider: {llm_provider}")
    print(f"Model: {model}")

    # Load data
    print(f"\nLoading data from: {NOTES_FILE_PATH}")
    df = pd.read_csv(NOTES_FILE_PATH)
    print(f"Loaded {len(df)} notes")

    if NOTES_COL not in df.columns:
        raise ValueError(f"Notes column '{NOTES_COL}' not found in data")

    # Extract features
    features_df = extract_all_features(
        df=df,
        notes_col=NOTES_COL,
        llm_provider=llm_provider,
        model=model
    )

    # Calculate derived features (30 additional features)
    print("\nCalculating derived features...")
    features_df = calculate_derived_features(features_df)
    print(f"Total features (base + derived): {features_df.shape[1]}")

    # Combine with original DataFrame to preserve INDEX_COL and row order
    if INDEX_COL:
        result_df = df[[INDEX_COL]].copy()
    else:
        result_df = pd.DataFrame()

    # Add all extracted features
    for col in features_df.columns:
        result_df[col] = features_df[col]

    # Save results
    output_dir = "saved_data"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "fixed_schema_features.csv")
    result_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")
    print(f"Output shape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)[:5]}... ({len(result_df.columns)} total)")

    print("\n" + "=" * 80)
    print("FIXED-SCHEMA EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nTo use with evaluation, add to config/evaluation_config.py:")
    print(f"ADDITIONAL_FEATURES = {{")
    print(f"    'fixed_schema': '{output_file}',")
    print(f"}}")


if __name__ == "__main__":
    main()
