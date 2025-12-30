"""
Feature operations including proposal, alignment, extraction, validation, and aggregation.
"""

import json
import math
from typing import Any, Dict, List
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import OutputParserException

from .data_utils import extract_json_from_tags
from config.prompts import FEATURE_PROPOSAL_TEMPLATE, FEATURE_ALIGNMENT_TEMPLATE, MERGE_FEATURE_TEMPLATE, FEATURE_VALIDATION_TEMPLATE, FEATURE_EXTRACTION_TEMPLATE, AGGREGATION_CODE_TEMPLATE
from core.log_utils import print

parser = JsonOutputParser()


# ============================================================================
# Feature Proposal and Confirmation
# ============================================================================

def propose_features(llm, notes_description, outcome_description,
                     structured_features_description, max_note_index):
    """
    Propose features from clinical notes using LLM with tool support.

    Args:
        llm: LLM tool caller instance
        notes_description: Description of the clinical notes
        outcome_description: Description of the prediction target
        structured_features_description: Description of existing structured features
        max_note_index: Maximum note index (0-based)

    Returns:
        List of proposed feature dictionaries
    """
    prompt = FEATURE_PROPOSAL_TEMPLATE.format(
        notes_description=notes_description,
        outcome_description=outcome_description,
        structured_feature_names=structured_features_description,
        MAX_NOTE_INDEX=max_note_index
    )

    while True:
        try:
            response = llm.query_with_tools(
                prompt,
                "Please use the note to validate that the features are structured enough to extract.",
                True
            )
            result = parser.parse(extract_json_from_tags(response))
            return result
        except OutputParserException as e:
            print(f"OutputParserException: {str(e)}")
            if "invalid json output" in str(e).lower():
                print("Invalid JSON format, retrying...")
                continue
            elif "Could not parse" in str(e).lower():
                print("Could not parse output, retrying...")
                continue
            else:
                print(f"Parser error: {e}, retrying...")
                continue
        except ValueError as e:
            if "Opening <JSON> tag not found" in str(e):
                print("Opening <JSON> tag not found")
                continue
            elif "Closing </JSON> tag not found" in str(e):
                print("Closing </JSON> tag not found")
                continue
            elif "No JSON content found" in str(e):
                print("No JSON content found in <JSON> tags or markdown code blocks")
                continue
            else:
                print("ValueError")
                raise e
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise e


def align_features(notes_list, features, query_llm_func, api_config,
                    notes_description, outcome_description,
                    structured_features_description, batch_size,
                    chunk_idx=None, batch_idx=None):
    """
    Confirm and validate features by reviewing them against actual notes.

    Args:
        notes_list: List of clinical notes to review
        features: List of feature dictionaries to align
        query_llm_func: LLM query function
        api_config: API configuration object
        notes_description: Description of the clinical notes
        outcome_description: Description of the prediction target
        structured_features_description: Description of existing structured features
        batch_size: Number of notes per batch
        chunk_idx: Optional chunk index for detailed logging
        batch_idx: Optional batch index for detailed logging

    Returns:
        List of aligned feature dictionaries with status information
    """
    notes = notes_list + [""] * (batch_size - len(notes_list)) if len(notes_list) < batch_size else notes_list[:batch_size]

    prompt = FEATURE_ALIGNMENT_TEMPLATE.format(
        notes_description=notes_description,
        outcome_description=outcome_description,
        structured_feature_names=structured_features_description,
        note_1=notes[0],
        note_2=notes[1],
        note_3=notes[2],
        note_4=notes[3],
        note_5=notes[4],
        note_6=notes[5],
        note_7=notes[6],
        note_8=notes[7],
        note_9=notes[8],
        note_10=notes[9],
        features=features
    )

    while True:
        try:
            full_response = query_llm_func(prompt, api_config, thinking=True, return_full_response=True)

            # Handle case where query_llm returns string instead of dict (shouldn't happen but defensive)
            if isinstance(full_response, str):
                print(f"Warning: query_llm returned string instead of dict, wrapping it")
                response = full_response
                full_response = {'text': full_response, 'thinking': None, 'raw': None}
            else:
                response = full_response['text']

            # Log detailed output if enabled (including thinking trace)
            if chunk_idx is not None and batch_idx is not None:
                try:
                    from core.log_utils import log_align_features_query
                    log_align_features_query(full_response, chunk_idx, batch_idx)
                except Exception as log_error:
                    print(f"Warning: Failed to log detailed output: {log_error}")

            result = parser.parse(extract_json_from_tags(response))
            return result
        except OutputParserException as e:
            print(f"OutputParserException: {str(e)}")
            if "invalid json output" in str(e).lower():
                print("Invalid JSON format, retrying...")
                continue
            elif "could not parse" in str(e).lower():
                print("Could not parse output, retrying...")
                continue
            else:
                print(f"Parser error: {e}, retrying...")
                continue
        except ValueError as e:
            if "Opening <JSON> tag not found" in str(e):
                print(f"JSON tag not found, regenerating...")
                continue
            elif "Closing </JSON> tag not found" in str(e):
                print(f"JSON tag not found, regenerating...")
                continue
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise e

def merge_features(notes_description: str, outcome_description: str, features,
                   query_llm_func, api_config):
    """
    Merge features from multiple chunks into a single unified feature set.

    Args:
        notes_description: Description of the clinical notes
        outcome_description: Description of the prediction target
        features: List of feature sets from different chunks (any number of chunks)
        query_llm_func: Function to query LLM
        api_config: API configuration object

    Returns:
        Unified list of features
    """
    num_chunks = len(features)

    # Build the feature sets text for any number of chunks
    feature_sets_text = ""
    for i, feature_set in enumerate(features, start=1):
        feature_sets_text += f"Feature set #{i}:\n{feature_set}\n\n"

    # Use the template from prompts.py
    prompt = MERGE_FEATURE_TEMPLATE.format(
        num_chunks=num_chunks,
        notes_description=notes_description,
        outcome_description=outcome_description,
        feature_sets=feature_sets_text
    )

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Query LLM with thinking mode and get full response
            full_response = query_llm_func(prompt, api_config, thinking=True, return_full_response=True)
            response = full_response['text']

            # Log the query (prompt, response, and thinking trace)
            from core.log_utils import log_merge_features_query
            log_merge_features_query(prompt, full_response, num_chunks)

            # Parse the JSON response
            result = parser.parse(extract_json_from_tags(response))
            return result
        except OutputParserException as e:
            retry_count += 1
            print(f"OutputParserException (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count >= max_retries:
                print(f"Failed after {max_retries} attempts")
                raise
            if "invalid json output" in str(e).lower():
                print("Invalid JSON format, retrying...")
                continue
            elif "could not parse" in str(e).lower():
                print("Could not parse output, retrying...")
                continue
            else:
                print(f"Parser error: {e}, retrying...")
                continue
        except ValueError as e:
            retry_count += 1
            print(f"ValueError (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count >= max_retries:
                print(f"Failed after {max_retries} attempts")
                raise
            if "Opening <JSON> tag not found" in str(e):
                print(f"JSON tag not found, regenerating...")
                continue
            elif "Closing </JSON> tag not found" in str(e):
                print(f"JSON tag not found, regenerating...")
                continue
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise e

    # If we exit the loop without returning, it means we exceeded max_retries
    raise RuntimeError(f"merge_features failed: exceeded {max_retries} retry attempts without successful parsing")


def combine_features_status(raw_features, status_features):
    """
    Combine raw_features with status-updated features from alignment process.

    Args:
        raw_features (list): Original list of feature dictionaries
        status_features (list): Feature updates with status information

    Returns:
        list: Combined list of complete feature dictionaries
    """
    # Create lookup for raw features by name
    raw_features_dict = {f['feature_name']: f for f in raw_features}

    # Create lookup for status features by name
    status_dict = {f['feature_name']: f for f in status_features if 'feature_name' in f}

    combined_features = []

    # Track changes for logging
    dropped_features = []
    new_features = []
    edited_features = []

    for feature_name, raw_feature in raw_features_dict.items():
        if feature_name in status_dict:
            status_feature = status_dict[feature_name]
            status = status_feature.get('status', 'aligned')

            if status == 'drop':
                # Keep dropped features but mark them as dropped
                dropped_feature = raw_feature.copy()
                dropped_feature['status'] = 'dropped'
                combined_features.append(dropped_feature)

                # Log dropped feature
                rationale = status_feature.get('rationale', 'No rationale provided')
                dropped_features.append({
                    'name': feature_name,
                    'rationale': rationale
                })

            elif status == 'aligned':
                # Keep original feature as-is (remove status if present)
                clean_feature = {k: v for k, v in raw_feature.items() if k != 'status'}
                combined_features.append(clean_feature)

            elif status == 'edited':
                # Track what was edited
                edits = {}

                # Combine updated fields into raw feature
                combined_feature = raw_feature.copy()
                for key, value in status_feature.items():
                    if key not in ['feature_name', 'status']:
                        # Check if this field was changed
                        if key not in raw_feature or raw_feature[key] != value:
                            edits[key] = {
                                'old': raw_feature.get(key, '[not present]'),
                                'new': value
                            }
                        combined_feature[key] = value

                # Remove status field
                clean_feature = {k: v for k, v in combined_feature.items() if k != 'status'}
                combined_features.append(clean_feature)

                # Log edited feature
                if edits:
                    edited_features.append({
                        'name': feature_name,
                        'changes': edits
                    })

            elif status == 'new':
                # This shouldn't happen for existing features, but handle gracefully
                print(f"WARNING: Existing feature '{feature_name}' is marked as new!")
                clean_feature = {k: v for k, v in status_feature.items() if k != 'status'}
                combined_features.append(clean_feature)
        else:
            # Feature not in status list - keep as confirmed
            print(f"INFO: Feature '{raw_feature['feature_name']}' not in status list, keeping as aligned")
            clean_feature = {k: v for k, v in raw_feature.items() if k != 'status'}
            combined_features.append(clean_feature)

    # Add truly new features from status_features
    for idx, status_feature in enumerate(status_features):
        # Validate that status_feature has required 'feature_name' key
        if 'feature_name' not in status_feature:
            print(f"ERROR: status_feature at index {idx} is missing 'feature_name' key. Feature content: {json.dumps(status_feature, indent=2)}")
            print("Skipping this malformed feature...")
            continue

        if status_feature.get('status') == 'new' and status_feature['feature_name'] not in raw_features_dict:
            clean_feature = {k: v for k, v in status_feature.items() if k != 'status'}
            combined_features.append(clean_feature)

            # Log new feature with full schema
            new_features.append(clean_feature)

    # Print summary logs
    print("\n" + "="*80)
    print("FEATURE COMBINATION SUMMARY")
    print("="*80)

    # Log dropped features
    if dropped_features:
        print(f"\n📍 DROPPED FEATURES ({len(dropped_features)}):")
        for feat in dropped_features:
            print(f"  - {feat['name']}: {feat['rationale']}")

    # Log new features
    if new_features:
        print(f"\n✅ NEW FEATURES ({len(new_features)}):")
        for feat in new_features:
            print(f"\n  Feature: {feat['feature_name']}")
            print(f"  Description: {feat.get('description', 'N/A')}")
            print(f"  Instructions: {feat.get('instructions', 'N/A')}")
            if feat.get('specific_subgroups'):
                print(f"  Subgroups ({len(feat['specific_subgroups'])}): {', '.join(feat['specific_subgroups'])}")
            print(f"  Is Aggregated: {feat.get('is_aggregated', False)}")
            if feat.get('aggregated_from'):
                print(f"  Aggregated From: {json.dumps(feat['aggregated_from'])}")

    # Log edited features
    if edited_features:
        print(f"\n✏️  EDITED FEATURES ({len(edited_features)}):")
        for feat in edited_features:
            print(f"\n  Feature: {feat['name']}")
            for field, changes in feat['changes'].items():
                print(f"    {field}:")
                print(f"      Old: {str(changes['old'])}")
                print(f"      New: {str(changes['new'])}")

    # Summary statistics
    print(f"\n📊 SUMMARY:")
    print(f"  Total features after combination: {len(combined_features)}")
    print(f"  Dropped: {len(dropped_features)}")
    print(f"  New: {len(new_features)}")
    print(f"  Edited: {len(edited_features)}")
    print(f"  Confirmed: {len(raw_features) - len(dropped_features) - len(edited_features)}")
    print("="*80 + "\n")

    return combined_features

def align_features_in_chunks(chunk, chunk_idx, initial_features, notes_col, checkpoint_dir,
                               align_features_func, combine_features_func):
    """
    Process a chunk of rows, passing batches of notes to align_features.

    Args:
        chunk: List of (index, row) tuples
        chunk_idx: Index of the current chunk
        initial_features: Initial feature list to start with
        notes_col: Name of the column containing notes
        checkpoint_dir: Directory for checkpoint files
        align_features_func: Function to align features
        combine_features_func: Function to combine features with status

    Returns:
        Final features after processing all batches in the chunk
    """
    import os
    import json
    from datetime import datetime
    from config.SNOW_config import BATCH_SIZE

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if there's a saved state for this chunk
    chunk_checkpoint_file = os.path.join(checkpoint_dir, f"chunk_{chunk_idx}_progress.json")
    batch_features_file = os.path.join(checkpoint_dir, f"chunk_{chunk_idx}_features.json")

    current_features = initial_features
    start_batch = 0

    # Load checkpoint if it exists
    if os.path.exists(chunk_checkpoint_file) and os.path.exists(batch_features_file):
        with open(chunk_checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_batch = checkpoint['last_completed_batch'] + 1
        with open(batch_features_file, 'r') as f:
            current_features = json.load(f)
        print(f"Resuming chunk {chunk_idx} from batch {start_batch}")

    # Group chunk into sub-batches
    note_batches = [chunk[i:i + BATCH_SIZE] for i in range(0, len(chunk), BATCH_SIZE)]

    for batch_idx in range(start_batch, len(note_batches)):
        batch = note_batches[batch_idx]
        print(f"Chunk {chunk_idx}, Batch {batch_idx+1}/{len(note_batches)}")
        notes_list = []
        row_indices = []

        for row_index, row in batch:
            row_indices.append(row_index)
            notes_list.append(row[notes_col])

        # Use align_features with batch of notes, retry if None
        status_features = None
        retry_count = 0
        max_retries = 3

        while status_features is None and retry_count < max_retries:
            status_features = align_features_func(notes_list, current_features, chunk_idx=chunk_idx, batch_idx=batch_idx)
            if status_features is None:
                retry_count += 1
                print(f"Warning: align_features returned None, retrying... (attempt {retry_count}/{max_retries})")

        if status_features is None:
            print(f"ERROR: align_features returned None after {max_retries} attempts, skipping batch")
            continue

        current_features = combine_features_func(current_features, status_features)

        print(f"Row indices: {row_indices}, features updated")

        # Save checkpoint after each batch
        checkpoint = {
            'chunk_idx': chunk_idx,
            'last_completed_batch': batch_idx,
            'total_batches': len(note_batches),
            'timestamp': datetime.now().isoformat()
        }
        with open(chunk_checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        with open(batch_features_file, 'w') as f:
            json.dump(current_features, f, indent=2)

    # Note: Checkpoint files are NOT cleaned up here - they will be cleaned up
    # by cleanup_checkpoints() after ALL chunks are completed
    print(f"Chunk {chunk_idx} completed successfully")

    return current_features


# ============================================================================
# Feature Extraction and Validation
# ============================================================================

def validate_feature(feature_spec, feature_values, outcome_description, notes_description,
                     all_feature_names, MAX_NOTE_INDEX, llm, validation_count, extraction_count):
    """
    Validate a single feature extracted across all notes.
    Returns a decision: 'proceed', 'remove', or 'reextract' with optional feedback.

    Args:
        feature_spec: Feature specification dictionary
        feature_values: Dictionary of extracted values for this feature
        outcome_description: Description of prediction target
        notes_description: Description of clinical notes
        all_feature_names: List of all feature names
        MAX_NOTE_INDEX: Maximum note index (0-indexed)
        llm: LLM instance with tool support
        validation_count: Number of times this feature has been validated
        extraction_count: Number of times this feature has been extracted

    Returns:
        Dictionary with validation decision and optional feedback
    """
    # Calculate statistics
    # Count values that are None, null, empty string, or NaN
    null_count = 0
    for v in feature_values.values():
        if v is None or v == "" or v == "null" or (isinstance(v, float) and math.isnan(v)):
            null_count += 1

    total_notes = len(feature_values)
    missing_percent = round((null_count / total_notes) * 100, 2) if total_notes > 0 else 0

    prompt = FEATURE_VALIDATION_TEMPLATE.format(
        feature_detail=feature_spec,
        notes_description=notes_description,
        extracted_values=feature_values,
        outcome_description=outcome_description,
        missing_percent=missing_percent,
        all_feature_names=all_feature_names,
        validation_count=validation_count,
        extraction_count=extraction_count,
        MAX_NOTE_INDEX=MAX_NOTE_INDEX
    )

    # Keep trying until we get valid JSON
    feature_name = feature_spec.get("feature_name", "unknown")
    while True:
        try:
            response = llm.query_with_tools(
                prompt,
                "Please reference the notes as needed to validate the feature values. Do not call the same index repeatedly.",
                thinking=False,
                feature_name=feature_name,
                iteration=validation_count
            )

            # Note: Detailed logging of the complete tool conversation (including the final response)
            # is handled automatically by llm.query_with_tools() when DETAILED_LOGGING is enabled

            result = parser.parse(extract_json_from_tags(response))
            return result
        except OutputParserException as e:
            # Handle parser-specific errors first (before ValueError)
            print(f"OutputParserException: {str(e)}")
            if "invalid json output" in str(e).lower():
                print("Invalid JSON format, retrying...")
                continue
            elif "Could not parse" in str(e).lower():
                print("Could not parse output, retrying...")
                continue
            else:
                # Log the specific parser error and retry
                print(f"Parser error: {e}, retrying...")
                continue
        except ValueError as e:
            if "Opening <JSON> tag not found" in str(e):
                print("Opening <JSON> tag not found")
                continue
            elif "Closing </JSON> tag not found" in str(e):
                print("Closing </JSON> tag not found")
                continue
            elif "No JSON content found" in str(e):
                print("No JSON content found in <JSON> tags or markdown code blocks")
                continue
            else:
                print("ValueError")
                # If it's a different error, re-raise
                raise e
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise e


def extract_features(note: str, feature_specs: List[Dict[str, Any]], notes_description: str,
                     query_llm_func, api_config, row_idx: int = None):
    """
    Call LLM to extract all **non‑aggregated** features from one note.

    Args:
        note: Clinical note text
        feature_specs: List of feature specifications to extract
        notes_description: Description of clinical notes
        query_llm_func: LLM query function
        api_config: API configuration object
        row_idx: Optional row index (kept for backward compatibility, not used for logging)

    Returns:
        Dictionary of extracted feature values
    """
    expected_features = [f["feature_name"] for f in feature_specs]

    prompt = FEATURE_EXTRACTION_TEMPLATE.format(
        note=note,
        features_detail=feature_specs,
        notes_description=notes_description,
        feature_list=expected_features
    )

    while True:
        try:
            response = query_llm_func(prompt, api_config, thinking=False)

            # Note: Extraction responses are NOT logged to detailed logs to reduce log volume
            # Only validation responses are logged since they are fewer and more critical

            result = parser.parse(extract_json_from_tags(response))
        except OutputParserException as e:
            # Handle parser-specific errors first (before ValueError)
            print(f"OutputParserException: {str(e)}")
            if "invalid json output" in str(e).lower():
                print("Invalid JSON format, retrying...")
                continue
            elif "Could not parse" in str(e).lower():
                print("Could not parse output, retrying...")
                continue
            else:
                # Log the specific parser error and retry
                print(f"Parser error: {e}, retrying...")
                continue
        except ValueError as e:
            if "Opening <JSON> tag not found" in str(e):
                print("Opening <JSON> tag not found")
                continue
            elif "Closing </JSON> tag not found" in str(e):
                print("Closing </JSON> tag not found")
                continue
            elif "No JSON content found" in str(e):
                print("No JSON content found in <JSON> tags or markdown code blocks")
                continue
            else:
                print("ValueError")
                # If it's a different error, re-raise
                raise e
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise e

        # Check if all expected features are in the result
        if set(result.keys()) == set(expected_features):
            return result

        # If we're here, the result is missing some features or has extra ones
        # Continue trying until we get the correct format


def generate_aggregation_code(agg_feature: dict[str, Any], aggregated_from_features, query_llm_func, api_config) -> str:
    """Generate Python code for aggregating a feature."""
    from config.prompts import AGGREGATION_CODE_TEMPLATE

    prompt = AGGREGATION_CODE_TEMPLATE.format(
        aggregated_feature=agg_feature,
        aggregated_from_features=aggregated_from_features
    )

    print(prompt)

    # Get full response with thinking trace
    full_response = query_llm_func(prompt, api_config, thinking=True, return_full_response=True)
    response = full_response['text']

    # Log the query (prompt, response, and thinking trace)
    from core.log_utils import log_aggregation_code_query
    log_aggregation_code_query(prompt, full_response, agg_feature['feature_name'])

    return response


# ============================================================================
# Helper Functions for Extract/Validate Workflow
# ============================================================================

def extract_row_features(row_data, features_to_extract, notes_col, notes_description, query_llm_func, api_config):
    """
    Extract features for a single row.

    Args:
        row_data: Tuple of (row_index, row)
        features_to_extract: List of feature specifications to extract
        notes_col: Name of column containing clinical notes
        notes_description: Description of clinical notes
        query_llm_func: LLM query function
        api_config: API configuration object

    Returns:
        Tuple of (row_index, extracted_features_dict)
    """
    row_index, row = row_data
    print("start ", row_index)
    response = extract_features(row[notes_col], features_to_extract, notes_description, query_llm_func, api_config, row_idx=row_index)
    print("end ", row_index)
    return row_index, response


def validate_single_feature(feature_spec, final_df, df, outcome_description, notes_description,
                            all_feature_names, MAX_NOTE_INDEX, llm_with_tools, validation_counts, extraction_counts):
    """
    Validate a single feature.

    Args:
        feature_spec: Feature specification dictionary
        final_df: DataFrame with extracted features
        df: Original DataFrame
        outcome_description: Description of prediction target
        notes_description: Description of clinical notes
        all_feature_names: List of all feature names in the model
        MAX_NOTE_INDEX: Maximum note index (0-indexed, so total notes = MAX_NOTE_INDEX + 1)
        llm_with_tools: LLM instance with tool support
        validation_counts: Dictionary tracking validation counts per feature
        extraction_counts: Dictionary tracking extraction counts per feature

    Returns:
        Tuple of (feature_spec, validation_response)
    """
    from config.prompts import FEATURE_VALIDATION_TEMPLATE

    feature_name = feature_spec["feature_name"]
    print(feature_spec)

    # Get current counts for this feature
    validation_count = validation_counts.get(feature_name, 0)
    extraction_count = extraction_counts.get(feature_name, 1)

    # Get current values for validation
    feature_values = {str(i): final_df[feature_name][i] for i in range(len(df))}

    # Validate the feature
    response = validate_feature(feature_spec, feature_values, outcome_description, notes_description,
                                all_feature_names, MAX_NOTE_INDEX, llm_with_tools, validation_count,
                                extraction_count)
    print(f"Validation result for {feature_name}: {response}")

    return feature_spec, response