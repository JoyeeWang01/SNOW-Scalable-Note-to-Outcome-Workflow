"""
Complete Pipeline for Oncology Feature Extraction.

This script orchestrates the full multi-agent LLM feature extraction pipeline:
1. Feature Definition: Discover and align features from clinical notes
2. Extract & Validate: Extract feature values and validate them iteratively
3. Aggregation: Generate aggregation code and create final dataset

Each stage can be enabled/disabled via command-line flags.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Import modular components
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.SNOW_config import MAIN_RUNS_DIR, DETAILED_LOGGING


def setup_run_directory():
    """Create timestamped run directory and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(MAIN_RUNS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Create run-specific subdirectories
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    if DETAILED_LOGGING:
        os.makedirs(os.path.join(run_dir, "detailed_logs"), exist_ok=True)

    # Ensure shared directories exist (checkpoints and saved_data are shared in MAIN_RUNS_DIR)
    os.makedirs(os.path.join(MAIN_RUNS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(MAIN_RUNS_DIR, "saved_data"), exist_ok=True)

    return run_dir, timestamp


def setup_logging(run_dir):
    """Configure logging to use run-specific directory."""
    import logging

    log_dir = os.path.join(run_dir, "logs")
    log_filename = os.path.join(log_dir, 'pipeline_log.txt')

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging with run-specific log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting pipeline - Run directory: {run_dir}")
    logger.info(f"Log file: {log_filename}")
    logger.info("=" * 80)

    return log_filename




def create_run_summary(run_dir, timestamp, args, status, error=None):
    """Create a summary file for the pipeline run."""
    summary_file = os.path.join(run_dir, "run_summary.txt")

    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PIPELINE RUN SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Provider: {args.provider}\n")
        f.write(f"Stages executed:\n")

        if not args.skip_definition and not args.only_extraction and not args.only_aggregation:
            f.write("  - Stage 1: Feature Definition\n")
        if not args.skip_extraction and not args.only_definition and not args.only_aggregation:
            f.write("  - Stage 2: Extract & Validate\n")
        if not args.skip_aggregation and not args.only_definition and not args.only_extraction:
            f.write("  - Stage 3: Aggregation\n")

        f.write(f"\nRun directory: {run_dir}\n")
        f.write(f"Log file: {os.path.join(run_dir, 'logs', 'pipeline_log.txt')}\n")
        f.write("\n" + "="*80 + "\n")
        f.write(f"Status: {status}\n")

        if error:
            f.write(f"Error: {error}\n")

        f.write("="*80 + "\n")

    return summary_file


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description='Multi-Agent LLM Feature Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python SNOW_oncology_pipeline.py --provider claude

  # Run only stages 2 and 3
  python SNOW_oncology_pipeline.py --provider gemini --skip-definition

  # Run only stage 1
  python SNOW_oncology_pipeline.py --provider claude --only-definition

  # Force re-run stage 1 even if features exist
  python SNOW_oncology_pipeline.py --provider openai --force
        """
    )

    parser.add_argument(
        '--provider',
        type=str,
        default='claude',
        choices=['gemini', 'claude', 'openai'],
        help='LLM provider to use (default: claude)'
    )

    parser.add_argument(
        '--skip-definition',
        action='store_true',
        help='Skip feature definition stage (use existing features)'
    )

    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip extract/validate stage (use existing extracted data)'
    )

    parser.add_argument(
        '--skip-aggregation',
        action='store_true',
        help='Skip aggregation stage'
    )

    parser.add_argument(
        '--only-definition',
        action='store_true',
        help='Run only feature definition stage'
    )

    parser.add_argument(
        '--only-extraction',
        action='store_true',
        help='Run only extract/validate stage'
    )

    parser.add_argument(
        '--only-aggregation',
        action='store_true',
        help='Run only aggregation stage'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run stages even if output exists'
    )

    args = parser.parse_args()

    # Create run directory
    run_dir, timestamp = setup_run_directory()

    # Setup logging
    log_filename = setup_logging(run_dir)

    print("="*80)
    print("MULTI-AGENT LLM FEATURE EXTRACTION PIPELINE")
    print("="*80)
    print(f"Run directory: {run_dir}")
    print(f"Log file: {log_filename}")
    print(f"Provider: {args.provider}")
    print("="*80)

    # Determine which stages to run
    run_stage_1 = not args.skip_definition
    run_stage_2 = not args.skip_extraction
    run_stage_3 = not args.skip_aggregation

    if args.only_definition:
        run_stage_1, run_stage_2, run_stage_3 = True, False, False
    elif args.only_extraction:
        run_stage_1, run_stage_2, run_stage_3 = False, True, False
    elif args.only_aggregation:
        run_stage_1, run_stage_2, run_stage_3 = False, False, True

    # Run stages
    try:
        if run_stage_1:
            print("\n" + "="*80)
            print("STAGE 1: FEATURE DEFINITION")
            print("="*80)

            # Import and run feature definition
            from SNOW_oncology_feature_definition import main as feature_definition_main
            feature_definition_main(provider=args.provider, run_dir=run_dir)

            print("\n" + "="*80)
            print("STAGE 1 COMPLETE")
            print("="*80)

        if run_stage_2:
            print("\n" + "="*80)
            print("STAGE 2: EXTRACT AND VALIDATE")
            print("="*80)

            # Import and run extract/validate
            from SNOW_oncology_extract_validate_loop import main as extract_validate_main
            extract_validate_main(provider=args.provider, run_dir=run_dir)

            print("\n" + "="*80)
            print("STAGE 2 COMPLETE")
            print("="*80)

        if run_stage_3:
            print("\n" + "="*80)
            print("STAGE 3: AGGREGATION")
            print("="*80)

            # Import and run aggregation
            from SNOW_oncology_aggregator import main as aggregation_main
            aggregation_main(provider=args.provider, run_dir=run_dir)

            print("\n" + "="*80)
            print("STAGE 3 COMPLETE")
            print("="*80)

        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\nRun directory: {run_dir}")
        print(f"  - Logs: {os.path.join(run_dir, 'logs')}")
        print(f"\nShared outputs (across all runs):")
        print(f"  - Checkpoints: {os.path.join(MAIN_RUNS_DIR, 'checkpoints')}")
        print(f"  - Saved features: {os.path.join(MAIN_RUNS_DIR, 'saved_data')}")

        # Check if detailed logs exist
        detailed_log_dir = os.path.join(run_dir, 'detailed_logs')
        if os.path.exists(detailed_log_dir):
            print(f"  - Detailed logs: {detailed_log_dir}")

        # Create summary
        summary_file = create_run_summary(run_dir, timestamp, args, "COMPLETED SUCCESSFULLY")
        print(f"\nRun summary saved to: {summary_file}")

        return 0

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        summary_file = create_run_summary(run_dir, timestamp, args, "INTERRUPTED BY USER")
        print(f"\nRun summary saved to: {summary_file}")
        return 130

    except Exception as e:
        print(f"\n\nERROR: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()

        summary_file = create_run_summary(run_dir, timestamp, args, "FAILED", error=str(e))
        print(f"\nRun summary saved to: {summary_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
