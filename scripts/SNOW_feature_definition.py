"""
This script handles the proposal and alignment of predictive features from
clinical notes using multi-agent LLM approach.
"""

from __future__ import annotations

import os
import json
import time
import pandas as pd
from functools import partial
import concurrent.futures

# Import modular components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.log_utils import setup_logging, print
# Note: Logging will be set up after run directory is created

from core.llm_interface import load_api_config, query_llm, ToolDefinition, LLMToolCaller
from config.SNOW_config import (
    NOTES_FILE_PATH,
    NOTES_COL,
    STRUCTURED_FEATURES_DESCRIPTION,
    NOTES_DESCRIPTION,
    OUTCOME_DESCRIPTION,
    NUM_CHUNKS,
    DETAILED_LOGGING,
    PROPOSE_NEW_FEATURES,
    PROCESS_CHUNKS_IN_PARALLEL,
    PARALLEL_THREAD_DELAY
)

# Checkpoint directory (overridden by pipeline to run-specific directory)
CHECKPOINT_DIR = "checkpoints"
from core.file_io import (
    load_features_from_file,
    save_features
)
from core.feature_operations import (
    propose_features,
    align_features,
    combine_features_status,
    align_features_in_chunks,
    merge_features
)
from core.checkpoint_manager import (
    load_aligned_feature_chunks,
    save_align_feature_progress,
    mark_chunk_completed,
    cleanup_checkpoints
)

# ============================================================================
# Main Function
# ============================================================================

def main(provider: str = "claude", run_dir: str = None):
    """
    Main feature definition function.

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
        log_file = setup_run_logging(log_dir, "feature_definition")
        print(f"Run directory: {run_dir}")
        print(f"Checkpoint directory (shared): {checkpoint_dir}")
        print(f"Log file: {log_file}")
    else:
        # Pipeline provided run_dir
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
            FEATURE_PROPOSAL_TEMPLATE,
            FEATURE_ALIGNMENT_TEMPLATE,
            MERGE_FEATURE_TEMPLATE
        )

        # Log all templates to a single file at startup
        with open(os.path.join(detailed_log_dir, "prompt_templates_feature_definition.txt"), 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE PROPOSAL TEMPLATE\n")
            f.write("=" * 80 + "\n\n")
            f.write(FEATURE_PROPOSAL_TEMPLATE)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("FEATURE ALIGNMENT TEMPLATE\n")
            f.write("=" * 80 + "\n\n")
            f.write(FEATURE_ALIGNMENT_TEMPLATE)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("MERGE FEATURE TEMPLATE\n")
            f.write("=" * 80 + "\n\n")
            f.write(MERGE_FEATURE_TEMPLATE)
        print(f"Detailed logging enabled: Prompt templates saved to {detailed_log_dir}/prompt_templates_feature_definition.txt")

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
    # Tool Setup for LLM
    # ============================================================================
    def get_note(index: int) -> str:
        """Get the full text of a clinical note by its index."""
        # Convert index to int if it's passed as string
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
    # Configuration: Choose whether to propose new features or load existing ones
    # ============================================================================
    # Note: These settings are now imported from config.pipeline_config

    # ============================================================================
    # Step 1: Proposing or Loading Features
    # ============================================================================
    if PROPOSE_NEW_FEATURES:
        print("Proposing new features from clinical notes...")
        raw_features = propose_features(
            llm_with_tools,
            NOTES_DESCRIPTION,
            OUTCOME_DESCRIPTION,
            STRUCTURED_FEATURES_DESCRIPTION,
            MAX_NOTE_INDEX
        )
        save_features(raw_features, run_dir=run_dir)
        print(f"Saved {len(raw_features)} proposed features")

        # Log raw features to main log if detailed logging is enabled
        if DETAILED_LOGGING:
            print("\n" + "="*80)
            print("RAW FEATURES (PROPOSED)")
            print("="*80)
            print(json.dumps(raw_features, indent=2))
            print("="*80 + "\n")
    else:
        print("Loading previously proposed features...")
        raw_features = load_features_from_file()
        print(f"Loaded {len(raw_features)} features")


    # ============================================================================
    # Step 2: Feature Alignment Process
    # ============================================================================
    # Create partial functions with pre-filled parameters
    align_features_partial = partial(
        align_features,
        query_llm_func=query_llm,
        api_config=api_config,
        notes_description=NOTES_DESCRIPTION,
        outcome_description=OUTCOME_DESCRIPTION,
        structured_features_description=STRUCTURED_FEATURES_DESCRIPTION,
        batch_size=10
    )

    # Prepare row data for chunked processing
    row_data_list = list(df.iterrows())
    total_rows = len(row_data_list)

    # Calculate chunk sizes to ensure exactly NUM_CHUNKS chunks
    chunk_sizes = [total_rows // NUM_CHUNKS] * NUM_CHUNKS
    # Distribute the remainder across the first few chunks
    for i in range(total_rows % NUM_CHUNKS):
        chunk_sizes[i] += 1

    # Create chunks with the calculated sizes
    row_chunks = []
    start = 0
    for size in chunk_sizes:
        row_chunks.append(row_data_list[start:start + size])
        start += size

    # Print chunk information
    print(f"Total rows: {total_rows}")
    print(f"Number of chunks: {len(row_chunks)}")
    for i, chunk in enumerate(row_chunks):
        print(f"Chunk {i}: {len(chunk)} rows")

    # Load completed chunks if any
    completed_chunks = load_aligned_feature_chunks(CHECKPOINT_DIR)

    final_features = []


    # ============================================================================
    # Helper Function for Processing Single Chunk
    # ============================================================================
    def process_single_chunk(chunk_idx, chunk, raw_features, completed_chunks):
        """
        Process a single chunk of rows for feature alignment.

        Args:
            chunk_idx: Index of the chunk
            chunk: List of (row_index, row) tuples
            raw_features: Initial features to align
            completed_chunks: List of already completed chunk indices

        Returns:
            Tuple of (chunk_idx, clean_features) or None if already completed
        """
        # Skip already completed chunks
        if chunk_idx in completed_chunks:
            # Load the saved features for this chunk
            chunk_features_file = os.path.join(CHECKPOINT_DIR, f"completed_chunk_{chunk_idx}.json")
            if os.path.exists(chunk_features_file):
                with open(chunk_features_file, 'r') as f:
                    clean_features = json.load(f)
                print(f"Skipping chunk {chunk_idx} (already completed with {len(clean_features)} features)")
                return (chunk_idx, clean_features, True)  # True indicates already completed

        print(f"\n{'='*50}")
        print(f"Processing chunk {chunk_idx}")
        print(f"{'='*50}")

        # Process the chunk with checkpoint support
        final_chunk_features = align_features_in_chunks(
            chunk,
            chunk_idx,
            initial_features=raw_features,
            notes_col=NOTES_COL,
            checkpoint_dir=CHECKPOINT_DIR,
            align_features_func=align_features_partial,
            combine_features_func=combine_features_status
        )

        # Remove features with status 'drop' and remove status field from remaining features
        clean_features = []
        for feature in final_chunk_features:
            if isinstance(feature, dict):
                # Skip features with status 'drop'
                if feature.get('status') == 'drop':
                    continue
                # Remove status field from remaining features
                clean_feature = {k: v for k, v in feature.items() if k != 'status'}
                clean_features.append(clean_feature)
            else:
                clean_features.append(feature)

        print(f"Completed chunk {chunk_idx} with {len(clean_features)} features")
        return (chunk_idx, clean_features, False)  # False indicates newly processed


    # ============================================================================
    # Step 3: Process Chunks with Checkpoint Support
    # ============================================================================
    if PROCESS_CHUNKS_IN_PARALLEL:
        print(f"\nProcessing {len(row_chunks)} chunks IN PARALLEL...")
        print(f"Staggering thread start by {PARALLEL_THREAD_DELAY} seconds to avoid API overload")
        print("Note: Progress updates may appear out of order due to parallel execution")

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CHUNKS) as executor:
            # Submit chunks with staggered delays to avoid API overload
            future_to_chunk = {}
            for chunk_idx, chunk in enumerate(row_chunks):
                future = executor.submit(process_single_chunk, chunk_idx, chunk, raw_features, completed_chunks)
                future_to_chunk[future] = chunk_idx

                # Add delay between submissions (except for the last one)
                if chunk_idx < len(row_chunks) - 1:
                    time.sleep(PARALLEL_THREAD_DELAY)

            # Collect results as they complete
            results = []
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    print(f"Chunk {chunk_idx} generated an exception: {exc}")
                    raise

            # Sort results by chunk_idx to maintain order
            results.sort(key=lambda x: x[0])

            # Update completed_chunks and final_features
            for chunk_idx, clean_features, was_already_completed in results:
                if not was_already_completed:
                    # Mark chunk as completed and update the list
                    completed_chunks = mark_chunk_completed(
                        CHECKPOINT_DIR,
                        chunk_idx,
                        completed_chunks,
                        clean_features
                    )
                final_features.append(clean_features)

    else:
        print(f"\nProcessing {len(row_chunks)} chunks SEQUENTIALLY...")

        # Sequential processing (original behavior)
        for chunk_idx, chunk in enumerate(row_chunks):
            result = process_single_chunk(chunk_idx, chunk, raw_features, completed_chunks)

            if result is not None:
                chunk_idx, clean_features, was_already_completed = result

                if not was_already_completed:
                    # Save overall progress
                    save_align_feature_progress(CHECKPOINT_DIR, len(row_chunks), chunk_idx, completed_chunks)

                    # Mark chunk as completed
                    completed_chunks = mark_chunk_completed(
                        CHECKPOINT_DIR,
                        chunk_idx,
                        completed_chunks,
                        clean_features
                    )

                final_features.append(clean_features)

    print(f"\n{'='*50}")
    print(f"All chunks completed successfully!")
    print(f"{'='*50}")

    # Clean up checkpoint files after successful completion
    cleanup_checkpoints(CHECKPOINT_DIR, completed_chunks, len(row_chunks))

    # ============================================================================
    # Step 4: Merge features from chunks
    # ============================================================================
    print(f"\n{'='*50}")
    print("Merging features from all chunks...")
    print(f"{'='*50}")

    # Merge features from all chunks into a single merged set
    current_features = merge_features(
        NOTES_DESCRIPTION,
        OUTCOME_DESCRIPTION,
        final_features,
        query_llm,
        api_config
    )

    print(f"Merged {len(current_features)} features from {len(final_features)} chunks")

    # Save the current features
    save_features(current_features, run_dir=run_dir)

    # Log final features to main log if detailed logging is enabled
    if DETAILED_LOGGING:
        print("\n" + "="*80)
        print("FINAL FEATURES (MERGED)")
        print("="*80)
        print(json.dumps(current_features, indent=2))
        print("="*80 + "\n")

    print("\n" + "="*80)
    print("FEATURE DEFINITION PROCESS COMPLETED")
    print("="*80)

# ============================================================================
# Standalone execution
# ============================================================================
if __name__ == "__main__":
    # Available providers: "gemini", "claude", "openai"
    SELECTED_PROVIDER = "claude"
    main(provider=SELECTED_PROVIDER)
