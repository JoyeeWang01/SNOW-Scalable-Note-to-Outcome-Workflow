"""
Utilities for managing run directories and paths.
"""

import os
from datetime import datetime
from config.SNOW_config import MAIN_RUNS_DIR, DETAILED_LOGGING


def setup_run_directory():
    """
    Create timestamped run directory and return paths.

    checkpoints and saved_data are shared across all runs (under MAIN_RUNS_DIR).
    logs and detailed_logs are run-specific.

    Returns:
        tuple: (run_dir, checkpoint_dir, log_dir, detailed_log_dir)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(MAIN_RUNS_DIR, f"run_{timestamp}")

    # Create main run directory
    os.makedirs(run_dir, exist_ok=True)

    # Shared directories (parallel to run folders)
    checkpoint_dir = os.path.join(MAIN_RUNS_DIR, "checkpoints")
    saved_data_dir = os.path.join(MAIN_RUNS_DIR, "saved_data")

    # Run-specific directories
    log_dir = os.path.join(run_dir, "logs")
    detailed_log_dir = os.path.join(run_dir, "detailed_logs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(saved_data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if DETAILED_LOGGING:
        os.makedirs(detailed_log_dir, exist_ok=True)

    return run_dir, checkpoint_dir, log_dir, detailed_log_dir


def setup_run_logging(log_dir, script_name):
    """
    Configure logging to use run-specific directory.

    Args:
        log_dir: Directory for log files
        script_name: Name of the script (for log filename)

    Returns:
        str: Path to log file
    """
    import logging

    log_filename = os.path.join(log_dir, f'{script_name}_log.txt')

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging with run-specific log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler()
        ]
    )

    return log_filename
