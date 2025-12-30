"""
This script handles the iterative extraction and validation of predictive features
from clinical notes using a multi-agent LLM approach with checkpointing.
"""

from __future__ import annotations

import os
import json
import time
import pandas as pd
import concurrent.futures
from datetime import datetime
from functools import partial

# Import modular components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.log_utils import setup_logging, print
# Note: Logging will be set up after run directory is created

from core.llm_interface import load_api_config, query_llm, ToolDefinition, LLMToolCaller
from core.data_utils import flatten_features
from core.feature_operations import extract_row_features, validate_single_feature
from core.file_io import load_features_from_file, save_df
from core.checkpoint_manager import (
    save_main_checkpoint,
    load_main_checkpoint,
    save_dataframe_checkpoint,
    load_dataframe_checkpoint,
    save_progress_checkpoint,
    load_progress_checkpoint,
    cleanup_progress_checkpoint,
    cleanup_all_checkpoints,
    check_checkpoint_exists
)

from config.SNOW_config import (
    NOTES_FILE_PATH,
    NOTES_COL,
    INDEX_COL,
    STRUCTURED_FEATURES_DESCRIPTION,
    NOTES_DESCRIPTION,
    OUTCOME_DESCRIPTION,
    MAX_WORKERS,
    EXTRACTION_CHECKPOINT_FREQUENCY,
    VALIDATION_CHECKPOINT_FREQUENCY,
    DETAILED_LOGGING,
    PARALLEL_THREAD_DELAY
)

# Checkpoint directory (overridden by workflow to run-specific directory)
CHECKPOINT_DIR = "checkpoints"

# ============================================================================
# Main Function
# ============================================================================

def main(provider: str = "claude", run_dir: str = None):
    """
    Main extract and validate function.

    Args:
        provider: LLM provider to use ("gemini", "claude", or "openai")
        run_dir: Optional run directory (if None, creates new timestamped directory)
    """
    # ============================================================================
    # Setup Run Directory
    # ============================================================================
    from core.run_utils import setup_run_directory, setup_run_logging

    if run_dir is None:
        run_dir, checkpoint_dir, log_dir, detailed_log_dir = setup_run_directory()
        log_file = setup_run_logging(log_dir, "extract_validate")
        print(f"Run directory: {run_dir}")
        print(f"Checkpoint directory (shared): {checkpoint_dir}")
        print(f"Log file: {log_file}")
    else:
        # Workflow provided run_dir
        from config.SNOW_config import MAIN_RUNS_DIR
        checkpoint_dir = os.path.join(MAIN_RUNS_DIR, "checkpoints")
        detailed_log_dir = os.path.join(run_dir, "detailed_logs")

    # Override global paths
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = checkpoint_dir
    import config.SNOW_config
    import core.log_utils
    config.SNOW_config.DETAILED_LOG_DIR = detailed_log_dir
    core.log_utils.DETAILED_LOG_DIR = detailed_log_dir

    # Log prompt templates if detailed logging is enabled
    if DETAILED_LOGGING:
        os.makedirs(detailed_log_dir, exist_ok=True)
        from config.prompts import (
            FEATURE_EXTRACTION_TEMPLATE,
            FEATURE_VALIDATION_TEMPLATE
        )

        # Log all templates to a single file at startup
        with open(os.path.join(detailed_log_dir, "prompt_templates_extract_validate_loop.txt"), 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE EXTRACTION TEMPLATE\n")
            f.write("=" * 80 + "\n\n")
            f.write(FEATURE_EXTRACTION_TEMPLATE)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("FEATURE VALIDATION TEMPLATE\n")
            f.write("=" * 80 + "\n\n")
            f.write(FEATURE_VALIDATION_TEMPLATE)
        print(f"Detailed logging enabled: Prompt templates saved to {detailed_log_dir}/prompt_templates_extract_validate_loop.txt")

    # ============================================================================
    # API Configuration
    # ============================================================================
    api_config = load_api_config(provider)

    # ============================================================================
    # Load and Prepare Dataset
    # ============================================================================
    df = pd.read_csv(NOTES_FILE_PATH)

    # Calculate the last valid index for notes
    MAX_NOTE_INDEX = len(df) - 1

    # Examine the dataframe
    print("Notes data shape:", df.shape)
    print("Notes data columns:", df.columns.tolist())
    print(f"Number of notes: {len(df)}")
    print(f"Max note index (0-indexed): {MAX_NOTE_INDEX}")

    # Extract notes from dataframe
    notes = df[NOTES_COL].tolist()

    # ============================================================================
    # Feature Definitions (Load from saved file)
    # ============================================================================
    print("Loading current features from saved file...")
    current_features = load_features_from_file()

    print(f"Number of current features: {len(current_features)}")

    # Log current features to main log if detailed logging is enabled
    if DETAILED_LOGGING:
        print("\n" + "="*80)
        print("CURRENT FEATURES (LOADED)")
        print("="*80)
        print(json.dumps(current_features, indent=2))
        print("="*80 + "\n")

    # Flatten features with subgroups
    current_features_flattened = flatten_features(current_features)

    # Split into aggregated vs non-aggregated
    aggregated = [f.copy() for f in current_features_flattened if f.get("is_aggregated", False)]
    non_aggregated = [f.copy() for f in current_features_flattened if not f.get("is_aggregated", False)]

    # Remove is_aggregated key
    for feature in aggregated + non_aggregated:
        feature.pop("is_aggregated", None)

    print(f"Loaded {len(non_aggregated)} non-aggregated and {len(aggregated)} aggregated features.")

    # ============================================================================
    # Tool Setup for LLM
    # ============================================================================
    def get_note(index: int) -> str:
        """Get the full text of a clinical note by its index."""
        if isinstance(index, str):
            index = int(index)
        return notes[index]

    # Tool definition for get_note function
    get_note_tool = ToolDefinition(
        name="get_note",
        description=f"Get the full text of a patient's clinical note by its index from 0 to {MAX_NOTE_INDEX}.",
        parameters={
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "The index of the note you would like to retrieve"
                }
            },
            "required": ["index"]
        },
        function=get_note
    )

    llm_with_tools = LLMToolCaller(
        api_config.url if hasattr(api_config, 'url') else None,
        api_config.key if hasattr(api_config, 'key') else None,
        api_config.model,
        [get_note_tool],
        api_config.llm_provider
    )

    # ============================================================================
    # Checkpoint Setup
    # ============================================================================
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Check what checkpoints exist
    has_main_checkpoint = check_checkpoint_exists(CHECKPOINT_DIR, "main_loop_checkpoint.json")
    has_df_checkpoint = check_checkpoint_exists(CHECKPOINT_DIR, "final_df_checkpoint.csv")

    print(f"Checkpoint status:")
    print(f"  Main checkpoint: {has_main_checkpoint}")
    print(f"  DataFrame checkpoint: {has_df_checkpoint}")

    # Initialize variables
    loop_iteration = 0
    features_to_reextract = []
    features_to_remove = []
    features_completed = []
    features_to_validate = []
    validation_counts = {}
    extraction_counts = {}

    # Load checkpoint if main checkpoint exists
    if has_main_checkpoint:
        checkpoint = load_main_checkpoint(CHECKPOINT_DIR)
        if checkpoint:
            loop_iteration = checkpoint.get('loop_iteration', 0)
            features_to_reextract = checkpoint.get('features_to_reextract', [])
            features_completed = checkpoint.get('features_completed', [])
            features_to_remove = checkpoint.get('features_to_remove', [])
            validation_counts = checkpoint.get('validation_counts', {})
            extraction_counts = checkpoint.get('extraction_counts', {})
            features_to_validate = checkpoint.get('features_to_validate', [])
            non_aggregated = checkpoint.get('non_aggregated', non_aggregated)
            print(f"Resuming from iteration {loop_iteration}")

    # Load DataFrame from checkpoint
    if has_df_checkpoint:
        final_df = load_dataframe_checkpoint(CHECKPOINT_DIR)
    else:
        # Only keep INDEX_COL and NOTES_COL from original dataframe
        cols_to_keep = [INDEX_COL, NOTES_COL]
        final_df = df[cols_to_keep].copy()
        print(f"Fresh start: Starting with fresh DataFrame (keeping only {cols_to_keep})")
        for feature_spec in non_aggregated:
            validation_counts[feature_spec["feature_name"]] = 0
            extraction_counts[feature_spec["feature_name"]] = 0

    # ============================================================================
    # Main Processing Loop
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"STARTING MULTI-AGENT ONCOLOGY PROCESSING")
    print(f"{'='*80}")
    print(f"LLM Provider: {api_config.llm_provider}")
    print(f"{'='*80}")

    while loop_iteration <= 5:
        print(f"\n{'='*50}")
        print(f"Loop iteration {loop_iteration}")
        print(f"{'='*50}")

        # Save checkpoint at start of each iteration
        checkpoint_data = {
            'loop_iteration': loop_iteration,
            'features_to_reextract': features_to_reextract,
            'features_completed': features_completed,
            'features_to_remove': features_to_remove,
            'features_to_validate': features_to_validate,
            'validation_counts': validation_counts,
            'extraction_counts': extraction_counts,
            'non_aggregated': non_aggregated
        }
        save_main_checkpoint(CHECKPOINT_DIR, checkpoint_data)

        # ========================================================================
        # Phase 1: Initial Extraction or Re-extraction
        # ========================================================================
        if not has_df_checkpoint:  # Fresh start - initial extraction
            print("\n" + "="*80)
            print("INITIAL EXTRACTION - STARTING")
            print("="*80)
            print(f"Features to extract: {len(non_aggregated)}")

            # Load or initialize extraction checkpoint
            extraction_checkpoint = load_progress_checkpoint(CHECKPOINT_DIR, "extraction_progress.json")
            extracted_features = extraction_checkpoint.get('extracted_features', {}) if extraction_checkpoint else {}
            completed_rows = set(extraction_checkpoint.get('completed_rows', [])) if extraction_checkpoint else set()

            if completed_rows:
                print(f"Resuming extraction - {len(completed_rows)} rows already completed")

            # Initialize extracted_features for all features
            for feature_spec in non_aggregated:
                if feature_spec["feature_name"] not in extracted_features:
                    extracted_features[feature_spec["feature_name"]] = [None] * len(df)

            # Prepare row data for parallel processing
            row_data_list = [(idx, row) for idx, row in df.iterrows() if idx not in completed_rows]

            if row_data_list:
                print(f"Processing {len(row_data_list)} remaining rows...")

                # Use ThreadPoolExecutor for parallel extraction
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    extract_func = partial(
                        extract_row_features,
                        features_to_extract=non_aggregated,
                        notes_col=NOTES_COL,
                        notes_description=NOTES_DESCRIPTION,
                        query_llm_func=query_llm,
                        api_config=api_config
                    )

                    # Progressive submission
                    completed_count = len(completed_rows)
                    future_to_row = {}
                    submitted = 0
                    max_concurrent = MAX_WORKERS

                    # Submit initial batch with staggered delays
                    print(f"Staggering thread start by {PARALLEL_THREAD_DELAY} seconds to avoid API overload")
                    while submitted < min(max_concurrent, len(row_data_list)):
                        row_data = row_data_list[submitted]
                        future = executor.submit(extract_func, row_data)
                        future_to_row[future] = row_data[0]
                        submitted += 1

                        # Add delay between submissions (except for the last one in batch)
                        if submitted < min(max_concurrent, len(row_data_list)):
                            time.sleep(PARALLEL_THREAD_DELAY)

                    # Process completions and submit new tasks
                    while future_to_row:
                        done, pending = concurrent.futures.wait(
                            future_to_row.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )

                        for future in done:
                            row_index, row_extracted = future.result()
                            for feature_name, value in row_extracted.items():
                                extracted_features[feature_name][row_index] = value

                            completed_rows.add(row_index)
                            completed_count += 1
                            del future_to_row[future]

                            # Submit next task if available
                            if submitted < len(row_data_list):
                                row_data = row_data_list[submitted]
                                new_future = executor.submit(extract_func, row_data)
                                future_to_row[new_future] = row_data[0]
                                submitted += 1

                            # Save checkpoint based on configured frequency
                            if completed_count % EXTRACTION_CHECKPOINT_FREQUENCY == 0 or not future_to_row:
                                progress_data = {
                                    'extracted_features': extracted_features,
                                    'completed_rows': list(completed_rows),
                                    'total_rows': len(df)
                                }
                                save_progress_checkpoint(CHECKPOINT_DIR, progress_data, "extraction_progress.json")
                                print(f"Extraction progress: {completed_count}/{len(df)} rows completed")

            # Clean up extraction checkpoint
            cleanup_progress_checkpoint(CHECKPOINT_DIR, "extraction_progress.json")

            # Update counts and prepare for validation
            features_to_validate = non_aggregated.copy()
            for feature_spec in features_to_validate:
                f_name = feature_spec["feature_name"]
                extraction_counts[f_name] = extraction_counts.get(f_name, 0) + 1

            # Add extracted features to final dataframe
            for feature_name, values in extracted_features.items():
                final_df[feature_name] = values

            # Save final_df
            save_df(final_df, run_dir=run_dir)
            save_dataframe_checkpoint(CHECKPOINT_DIR, final_df)

            has_df_checkpoint = True

            checkpoint_data['extraction_counts'] = extraction_counts
            checkpoint_data['features_to_validate'] = features_to_validate
            save_main_checkpoint(CHECKPOINT_DIR, checkpoint_data)

        elif features_to_reextract:  # Re-extraction phase
            print("\n" + "="*80)
            print("RE-EXTRACTION PHASE - STARTING")
            print("="*80)
            print(f"Features to re-extract: {len(features_to_reextract)}")

            # Similar logic to initial extraction but only for features_to_reextract
            reextraction_checkpoint = load_progress_checkpoint(CHECKPOINT_DIR, "reextraction_progress.json")
            extracted_features = reextraction_checkpoint.get('extracted_features', {}) if reextraction_checkpoint else {}
            completed_rows = set(reextraction_checkpoint.get('completed_rows', [])) if reextraction_checkpoint else set()

            for feature_spec in features_to_reextract:
                if feature_spec["feature_name"] not in extracted_features:
                    extracted_features[feature_spec["feature_name"]] = [None] * len(df)

            row_data_list = [(idx, row) for idx, row in df.iterrows() if idx not in completed_rows]

            if row_data_list:
                print(f"Processing {len(row_data_list)} remaining rows for re-extraction...")

                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    extract_func = partial(
                        extract_row_features,
                        features_to_extract=features_to_reextract,
                        notes_col=NOTES_COL,
                        notes_description=NOTES_DESCRIPTION,
                        query_llm_func=query_llm,
                        api_config=api_config
                    )

                    # Same progressive submission pattern with staggered delays
                    completed_count = len(completed_rows)
                    future_to_row = {}
                    submitted = 0

                    print(f"Staggering thread start by {PARALLEL_THREAD_DELAY} seconds to avoid API overload")
                    while submitted < min(MAX_WORKERS, len(row_data_list)):
                        row_data = row_data_list[submitted]
                        future = executor.submit(extract_func, row_data)
                        future_to_row[future] = row_data[0]
                        submitted += 1

                        # Add delay between submissions (except for the last one in batch)
                        if submitted < min(MAX_WORKERS, len(row_data_list)):
                            time.sleep(PARALLEL_THREAD_DELAY)

                    while future_to_row:
                        done, pending = concurrent.futures.wait(
                            future_to_row.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )

                        for future in done:
                            row_index, row_extracted = future.result()
                            for feature_name, value in row_extracted.items():
                                extracted_features[feature_name][row_index] = value

                            completed_rows.add(row_index)
                            completed_count += 1
                            del future_to_row[future]

                            if submitted < len(row_data_list):
                                row_data = row_data_list[submitted]
                                new_future = executor.submit(extract_func, row_data)
                                future_to_row[new_future] = row_data[0]
                                submitted += 1

                            # Save checkpoint based on configured frequency
                            if completed_count % EXTRACTION_CHECKPOINT_FREQUENCY == 0 or not future_to_row:
                                progress_data = {
                                    'extracted_features': extracted_features,
                                    'completed_rows': list(completed_rows),
                                    'total_rows': len(df)
                                }
                                save_progress_checkpoint(CHECKPOINT_DIR, progress_data, "reextraction_progress.json")
                                print(f"Re-extraction progress: {completed_count}/{len(df)} rows completed")

            cleanup_progress_checkpoint(CHECKPOINT_DIR, "reextraction_progress.json")

            # Update counts
            for feature_spec in features_to_reextract:
                f_name = feature_spec["feature_name"]
                extraction_counts[f_name] = extraction_counts.get(f_name, 0) + 1

            features_to_validate = features_to_reextract
            features_to_reextract = []

            # Update final dataframe
            for feature_name, values in extracted_features.items():
                final_df[feature_name] = values

            save_df(final_df, run_dir=run_dir)
            save_dataframe_checkpoint(CHECKPOINT_DIR, final_df)

            checkpoint_data['extraction_counts'] = extraction_counts
            checkpoint_data['features_to_validate'] = features_to_validate
            checkpoint_data['features_to_reextract'] = features_to_reextract
            save_main_checkpoint(CHECKPOINT_DIR, checkpoint_data)

        # ========================================================================
        # Phase 2: Validation
        # ========================================================================
        print("\n" + "="*80)
        print("VALIDATION PHASE - STARTING")
        print("="*80)
        print(f"Features to validate: {len(features_to_validate)}")

        # Load or initialize validation checkpoint
        validation_checkpoint = load_progress_checkpoint(CHECKPOINT_DIR, "validation_progress.json")
        validated_features = set(validation_checkpoint.get('validated_features', [])) if validation_checkpoint else set()

        if validation_checkpoint:
            features_completed = validation_checkpoint.get('features_completed', features_completed)
            features_to_remove = validation_checkpoint.get('features_to_remove', features_to_remove)
            features_to_reextract = validation_checkpoint.get('features_to_reextract', [])
            validation_counts = validation_checkpoint.get('validation_counts', validation_counts)
            print(f"Resuming validation - {len(validated_features)} features already validated")

        # Filter out already validated features
        features_to_validate_copy = [f for f in features_to_validate if f["feature_name"] not in validated_features]

        if features_to_validate_copy:
            print(f"Validating {len(features_to_validate_copy)} remaining features...")

            # Prepare all feature names for context (including structured features)
            feature_names_list = [f["feature_name"] for f in aggregated]
            feature_names_list.extend([f["feature_name"] for f in non_aggregated])
            all_feature_names = STRUCTURED_FEATURES_DESCRIPTION + ", " + ", ".join(feature_names_list)

            # Use ThreadPoolExecutor for parallel validation
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                validate_func = partial(
                    validate_single_feature,
                    final_df=final_df,
                    df=df,
                    outcome_description=OUTCOME_DESCRIPTION,
                    notes_description=NOTES_DESCRIPTION,
                    all_feature_names=all_feature_names,
                    MAX_NOTE_INDEX=MAX_NOTE_INDEX,
                    llm_with_tools=llm_with_tools,
                    validation_counts=validation_counts,
                    extraction_counts=extraction_counts
                )

                # Progressive submission with staggered delays
                completed_count = len(validated_features)
                future_to_feature = {}
                submitted = 0

                print(f"Staggering validation thread start by {PARALLEL_THREAD_DELAY} seconds to avoid API overload")
                while submitted < min(MAX_WORKERS, len(features_to_validate_copy)):
                    feature_spec = features_to_validate_copy[submitted]
                    future = executor.submit(validate_func, feature_spec)
                    future_to_feature[future] = feature_spec
                    submitted += 1

                    # Add delay between submissions (except for the last one in batch)
                    if submitted < min(MAX_WORKERS, len(features_to_validate_copy)):
                        time.sleep(PARALLEL_THREAD_DELAY)

                processed = 0
                while future_to_feature:
                    done, pending = concurrent.futures.wait(
                        future_to_feature.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        feature_spec, response = future.result()
                        feature_name = feature_spec["feature_name"]

                        validated_features.add(feature_name)
                        completed_count += 1
                        processed += 1

                        # Handle new features
                        new_features = response.get("add_additional_feature")
                        if new_features:
                            new_features = flatten_features(new_features) if isinstance(new_features, list) else [new_features]

                            # Get existing feature names from both non_aggregated and aggregated
                            existing_names = {f["feature_name"] for f in non_aggregated + aggregated}

                            # Check each feature and only add if it doesn't already exist
                            added = []
                            skipped = []
                            for new_f in new_features:
                                fname = new_f["feature_name"]
                                if fname not in existing_names:
                                    non_aggregated.append(new_f)
                                    features_to_reextract.append(new_f)
                                    added.append(fname)
                                    existing_names.add(fname)  # Add to set to catch duplicates within new_features
                                else:
                                    skipped.append(fname)

                            if added:
                                print(f"Adding {len(added)} new features: {', '.join(added)}")
                            if skipped:
                                print(f"Skipping {len(skipped)} duplicate features: {', '.join(skipped)}")

                        # Handle validation decision
                        if response["decision"] == "proceed":
                            features_completed.append(feature_spec)
                            print(f"Feature {feature_name} completed")
                        elif response["decision"] == "remove":
                            features_to_remove.append(feature_name)
                            print(f"Feature {feature_name} marked for removal")
                        elif response["decision"] == "reextract":
                            if validation_counts.get(feature_name, 0) >= 3:
                                features_to_remove.append(feature_name)
                                print(f"Feature {feature_name} marked for removal (exceeded 3 attempts)")
                            else:
                                feature_spec["instructions"] = response.get("current_feature_instructions", feature_spec.get("instructions"))
                                features_to_reextract.append(feature_spec)
                                print(f"Feature {feature_name} marked for re-extraction")

                        del future_to_feature[future]

                        # Submit next task
                        if submitted < len(features_to_validate_copy):
                            next_feature_spec = features_to_validate_copy[submitted]
                            new_future = executor.submit(validate_func, next_feature_spec)
                            future_to_feature[new_future] = next_feature_spec
                            submitted += 1

                        # Save checkpoint based on configured frequency
                        if completed_count % VALIDATION_CHECKPOINT_FREQUENCY == 0 or not future_to_feature:
                            progress_data = {
                                'validated_features': list(validated_features),
                                'features_completed': features_completed,
                                'features_to_remove': features_to_remove,
                                'features_to_reextract': features_to_reextract,
                                'validation_counts': validation_counts,
                                'total_features': len(features_to_validate)
                            }
                            save_progress_checkpoint(CHECKPOINT_DIR, progress_data, "validation_progress.json")
                            print(f"Validation progress: {completed_count}/{len(features_to_validate)} features validated")

        cleanup_progress_checkpoint(CHECKPOINT_DIR, "validation_progress.json")

        # Increment validation counts
        for feature_spec in features_to_validate:
            f_name = feature_spec["feature_name"]
            validation_counts[f_name] = validation_counts.get(f_name, 0) + 1

        features_to_validate = []

        # Save checkpoint and dataframe
        checkpoint_data['features_to_reextract'] = features_to_reextract
        checkpoint_data['features_completed'] = features_completed
        checkpoint_data['features_to_remove'] = features_to_remove
        checkpoint_data['validation_counts'] = validation_counts
        checkpoint_data['extraction_counts'] = extraction_counts
        checkpoint_data['non_aggregated'] = non_aggregated
        save_main_checkpoint(CHECKPOINT_DIR, checkpoint_data)
        save_dataframe_checkpoint(CHECKPOINT_DIR, final_df)

        # Check if we're done
        if not features_to_reextract:
            print("All features processed successfully. Breaking loop.")
            cleanup_all_checkpoints(CHECKPOINT_DIR)
            break

        print(f"\nNext iteration: {len(features_to_reextract)} features to reextract")
        print(f"  Features completed: {len(features_completed)}")
        print(f"  Features removed: {len(features_to_remove)}")
        loop_iteration += 1

    # ============================================================================
    # Final Processing
    # ============================================================================
    print("\n" + "="*80)
    print("FINAL PROCESSING")
    print("="*80)

    # Remove features marked for removal
    for feature_name in features_to_remove:
        if feature_name in final_df.columns:
            final_df = final_df.drop(columns=[feature_name])
            print(f"Removed feature: {feature_name}")

    # Remove features marked for removal from non_aggregated list
    non_aggregated_filtered = [f for f in non_aggregated if f["feature_name"] not in features_to_remove]

    # Add is_aggregated field to all features
    for feature in non_aggregated_filtered:
        feature["is_aggregated"] = False

    for feature in aggregated:
        feature["is_aggregated"] = True

    # Combine non-aggregated and aggregated features
    final_features = non_aggregated_filtered + aggregated

    # Save the final features
    from core.file_io import save_features
    save_features(final_features, run_dir=run_dir)
    print(f"Saved {len(final_features)} features ({len(non_aggregated_filtered)} non-aggregated, {len(aggregated)} aggregated)")

    # Log final features to main log if detailed logging is enabled
    if DETAILED_LOGGING:
        print("\n" + "="*80)
        print("FINAL FEATURES (AFTER VALIDATION)")
        print("="*80)
        print(json.dumps(final_features, indent=2))
        print("="*80 + "\n")

    print(f"\nProcessing complete!")
    print(f"Final dataframe has {len(final_df.columns)} columns")
    print(f"Features completed: {len(features_completed)}")
    print(f"Features removed: {len(features_to_remove)}")

    # Save final_df
    save_df(final_df, run_dir=run_dir)
    save_dataframe_checkpoint(CHECKPOINT_DIR, final_df)

    print("\n" + "="*80)
    print("FEATURE EXTRACTION AND VALIDATION COMPLETED")
    print("="*80)

# ============================================================================
# Standalone execution
# ============================================================================
if __name__ == "__main__":
    # Available providers: "gemini", "claude", "openai"
    SELECTED_PROVIDER = "claude"
    main(provider=SELECTED_PROVIDER)
