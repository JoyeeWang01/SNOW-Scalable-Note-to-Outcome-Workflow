from __future__ import annotations

import os
import json
import pandas as pd
from typing import Any

# Import modular components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.log_utils import setup_logging, print
# Note: Logging will be set up after run directory is created

from core.llm_interface import load_api_config
from config.SNOW_config import DETAILED_LOGGING
from core.feature_operations import generate_aggregation_code
from core.file_io import load_df_from_file, load_features_from_file
from core.llm_interface import query_llm
from core.data_utils import flatten_features


def main(provider: str = "claude", run_dir: str = None):
    """
    Main aggregation function.

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
        log_file = setup_run_logging(log_dir, "aggregator")
        print(f"Run directory: {run_dir}")
        print(f"Log file: {log_file}")
    else:
        # Pipeline provided run_dir
        from config.SNOW_config import MAIN_RUNS_DIR
        checkpoint_dir = os.path.join(MAIN_RUNS_DIR, "checkpoints")
        detailed_log_dir = os.path.join(run_dir, "detailed_logs")

    # Override global paths
    import config.SNOW_config
    import core.log_utils
    config.SNOW_config.DETAILED_LOG_DIR = detailed_log_dir
    core.log_utils.DETAILED_LOG_DIR = detailed_log_dir

    # Log prompt templates if detailed logging is enabled
    if DETAILED_LOGGING:
        os.makedirs(detailed_log_dir, exist_ok=True)
        from config.prompts import AGGREGATION_CODE_TEMPLATE

        # Log all templates to a single file at startup
        with open(os.path.join(detailed_log_dir, "prompt_templates_aggregator.txt"), 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AGGREGATION CODE TEMPLATE\n")
            f.write("=" * 80 + "\n\n")
            f.write(AGGREGATION_CODE_TEMPLATE)
        print(f"Detailed logging enabled: Prompt templates saved to {detailed_log_dir}/prompt_templates_aggregator.txt")

    # ============================================================================
    # API Configuration
    # ============================================================================
    api_config = load_api_config(provider)

    # ============================================================================
    # Feature Definitions (Load from saved file)
    # ============================================================================
    print("Loading current features from saved file...")
    current_features = load_features_from_file(run_dir=run_dir)
    #feature list after extract/validate loop doesn't need to be flattened

    print(f"Number of current features: {len(current_features)}")

    # Log current features to main log if detailed logging is enabled
    if DETAILED_LOGGING:
        print("\n" + "="*80)
        print("CURRENT FEATURES (LOADED)")
        print("="*80)
        print(json.dumps(current_features, indent=2))
        print("="*80 + "\n")

    # Split into aggregated vs non-aggregated
    aggregated = [f.copy() for f in current_features if f.get("is_aggregated", False)]
    non_aggregated = [f.copy() for f in current_features if not f.get("is_aggregated", False)]

    # Remove is_aggregated key
    for feature in aggregated + non_aggregated:
        feature.pop("is_aggregated", None)

    print(f"Loaded {len(non_aggregated)} non-aggregated and {len(aggregated)} aggregated features.")

    # Create a lookup dictionary for non-aggregated feature specs by name
    non_aggregated_dict = {f["feature_name"]: f for f in non_aggregated}

    # Get list of present feature names (non-aggregated features that are in the data)
    features_present = [f["feature_name"] for f in non_aggregated]

    final_df = load_df_from_file(run_dir=run_dir)

    aggregated_features_to_remove = []

    for aggregated_feature in aggregated:
        if 'aggregated_from' in aggregated_feature:
            # Filter out any features that were removed (checking by feature name)
            # Note: aggregated_from contains strings (feature names), not dicts
            original_aggregated_from = aggregated_feature['aggregated_from'].copy()
            aggregated_feature['aggregated_from'] = [
                feature_name for feature_name in aggregated_feature['aggregated_from']
                if feature_name in features_present
            ]

            # Print if any changes were made
            if len(aggregated_feature['aggregated_from']) != len(original_aggregated_from):
                removed_from_aggregate = [
                    feature_name for feature_name in original_aggregated_from
                    if feature_name not in features_present
                ]
                print(f"Updated {aggregated_feature['feature_name']}: removed {removed_from_aggregate} from aggregated_from")

            # If aggregated_from has <= 1 features, mark this aggregated feature for removal
            if len(aggregated_feature['aggregated_from']) <= 1:
                aggregated_features_to_remove.append(aggregated_feature['feature_name'])
                print(f"Marking aggregated feature {aggregated_feature['feature_name']} for removal: only {len(aggregated_feature['aggregated_from'])} feature(s) remaining")

    # Remove aggregated features that have <= 1 aggregated_from features
    aggregated = [
        feature for feature in aggregated
        if feature['feature_name'] not in aggregated_features_to_remove
    ]

    print(f"Removed {len(aggregated_features_to_remove)} aggregated features with <= 1 aggregated_from features")

    # Load or initialize aggregation code checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, "aggregation_code_checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        agg_code_blocks = checkpoint.get('agg_code_blocks', [])
        completed_features = checkpoint.get('completed_features', [])
        completed_feature_names = {f['feature_name'] for f in completed_features}
        print(f"Loaded checkpoint: {len(completed_features)} features already completed")
    else:
        agg_code_blocks = []
        completed_features = []
        completed_feature_names = set()

    print(f"\nGenerating aggregation code for {len(aggregated)} features...")
    failed_features = []  # Track features that failed after retries

    for idx, agg in enumerate(aggregated, 1):
        feature_name = agg['feature_name']

        # Skip if already completed
        if feature_name in completed_feature_names:
            print(f"  [{idx}/{len(aggregated)}] Skipping (already completed): {feature_name}")
            continue

        print(f"  [{idx}/{len(aggregated)}] Generating code for: {feature_name}")

        # Get feature specs for all aggregated_from features
        source_feature_specs = [
            non_aggregated_dict[fname] for fname in agg['aggregated_from']
            if fname in non_aggregated_dict
        ]

        # Retry up to 3 times on error
        max_retries = 3
        code_str = None
        for attempt in range(1, max_retries + 1):
            code_str = generate_aggregation_code(agg, source_feature_specs, query_llm, api_config)
            if "<ERROR>" not in code_str:
                break  # Success

            if attempt < max_retries:
                print(f"    ⚠️  Attempt {attempt}/{max_retries} failed, retrying...")
            else:
                print(f"    ❌ Failed after {max_retries} attempts: {code_str}")
                failed_features.append(agg)

        # Only add to completed if successful
        if code_str and "<ERROR>" not in code_str:
            print(f"    ✓ Code generated successfully")
            agg_code_blocks.append(code_str)
            completed_features.append(agg)
            completed_feature_names.add(feature_name)

            # Save checkpoint after each successful generation
            checkpoint_data = {
                'agg_code_blocks': agg_code_blocks,
                'completed_features': completed_features
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

    print(f"\nCompleted: {len(completed_features)}/{len(aggregated)} features generated successfully")

    # Report failed features
    if failed_features:
        print(f"\n⚠️  {len(failed_features)} features failed after {max_retries} attempts:")
        for failed_agg in failed_features:
            print(f"  - {failed_agg['feature_name']}")

        # Remove failed features from aggregated list and save updated features
        print(f"\nRemoving {len(failed_features)} failed features from aggregated list...")
        updated_aggregated = [agg for agg in aggregated if agg not in failed_features]

        # Combine non_aggregated and updated_aggregated into single list
        updated_features = non_aggregated + updated_aggregated

        # Save updated feature list
        from core.file_io import save_features
        save_features(updated_features)
        print(f"✓ Updated features saved (removed {len(failed_features)} failed aggregated features)")
        print(f"  Final feature counts: {len(non_aggregated)} non-aggregated + {len(updated_aggregated)} aggregated = {len(updated_features)} total")
    else:
        # No failures - report original counts
        print(f"\nFinal feature counts: {len(non_aggregated)} non-aggregated + {len(aggregated)} aggregated = {len(non_aggregated) + len(aggregated)} total")

    # Clean up checkpoint file after completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Checkpoint cleaned up")

    # stitch them together + tiny wrapper so old API still works
    combined_code = "\n\n".join(agg_code_blocks) + """

AGGREGATORS = {
%s
}

def aggregate_features(features: dict[str, Any]) -> dict[str, Any]:
    return {name: fn(features) for name, fn in AGGREGATORS.items()}
""" % (
        "\n".join(f'    "{agg["feature_name"]}": aggregate_{agg["feature_name"].lower().replace(" ", "_")},'
                  for agg in completed_features)
    )

    agg_code = combined_code or "def aggregate_features(features):\n    return {}"

    # Check if the aggregation code contains ```python or ``` markers and remove them
    while "```python" in agg_code:
        agg_code = agg_code.replace("```python", "")
    while "```" in agg_code:
        agg_code = agg_code.replace("```", "")

    ns: dict[str, Any] = {}
    exec(agg_code, ns)          # same dict is used for globals *and* locals

    aggregate_fn  = ns["aggregate_features"]
    AGGREGATORS   = ns["AGGREGATORS"]

    records = []
    for _, row in final_df.iterrows():
        base = {col: row[col] for col in row.index}
        base.update(aggregate_fn(base))
        records.append(base)

    results_df = pd.DataFrame(records)

    # Remove NOTES_COL from final output (contains clinical text)
    from config.SNOW_config import NOTES_COL
    if NOTES_COL in results_df.columns:
        results_df = results_df.drop(columns=[NOTES_COL])
        print(f"Removed '{NOTES_COL}' column from final output")

    # Save the DataFrame to a CSV file with timestamp
    from datetime import datetime
    from core.file_io import get_saved_data_dir

    saved_data_dir = get_saved_data_dir(run_dir)
    timestamp = datetime.now().strftime("%m%d")
    output_file = os.path.join(saved_data_dir, f'generated_features_{timestamp}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Saved {len(results_df)} records to '{output_file}'")


# ============================================================================
# Standalone execution
# ============================================================================
if __name__ == "__main__":
    # Available providers: "gemini", "claude", "openai"
    SELECTED_PROVIDER = "claude"
    main(provider=SELECTED_PROVIDER)