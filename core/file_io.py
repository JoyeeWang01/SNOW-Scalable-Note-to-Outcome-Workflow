"""
File I/O operations for feature management.
"""

import json
import os
import pandas as pd
from datetime import datetime
from core.log_utils import print


def get_saved_data_dir(run_dir=None):
    """
    Get the saved_data directory path.

    saved_data is shared across all runs under MAIN_RUNS_DIR.
    If run_dir is None (standalone execution), uses current directory.

    Args:
        run_dir: Run directory from workflow (if provided, extracts MAIN_RUNS_DIR)

    Returns:
        Path to saved_data directory
    """
    if run_dir is None:
        return "saved_data"
    else:
        # Extract MAIN_RUNS_DIR from run_dir (e.g., "SNOW_runs/run_20251009_143105" -> "SNOW_runs")
        from config.SNOW_config import MAIN_RUNS_DIR
        return os.path.join(MAIN_RUNS_DIR, "saved_data")


def load_features_from_file(filename=None, run_dir=None):
    """
    Load raw_features from a saved file.

    Args:
        filename: Input filename (json). If None, uses features_latest.json from saved_data
        run_dir: Run directory (optional). If None and no filename provided, looks in MAIN_RUNS_DIR/saved_data

    Returns:
        List of feature dictionaries
    """
    if filename is None:
        # Default to shared saved_data location
        if run_dir is None:
            from config.SNOW_config import MAIN_RUNS_DIR
            saved_data_dir = os.path.join(MAIN_RUNS_DIR, "saved_data")
        else:
            saved_data_dir = get_saved_data_dir(run_dir)

        filename = os.path.join(saved_data_dir, "features_latest.json")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No saved features found at {filename}")

    with open(filename, 'r') as f:
        features = json.load(f)

    print(f"Loaded {len(features)} features from {filename}")
    return features


def save_features(features, run_dir=None):
    """
    Save raw features to both timestamped and latest files.

    Args:
        features: List of feature dictionaries
        run_dir: Run directory (optional, for workflow runs)
    """
    saved_data_dir = get_saved_data_dir(run_dir)

    # Create directory if needed
    os.makedirs(saved_data_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save features to JSON file with timestamp
    timestamped_filename = os.path.join(saved_data_dir, f"features_{timestamp}.json")
    with open(timestamped_filename, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"Saved raw_features to {timestamped_filename}")

    # Also save to a fixed filename for easy access to the latest version
    latest_filename = os.path.join(saved_data_dir, "features_latest.json")
    with open(latest_filename, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"Also saved as {latest_filename}")


def load_df_from_file(filename=None, run_dir=None):
    """
    Load dataframe from a saved CSV file.

    Args:
        filename: Input filename (csv). If None, loads dataframe_latest.csv from saved_data
        run_dir: Run directory (optional). If None and no filename provided, looks in MAIN_RUNS_DIR/saved_data

    Returns:
        pandas DataFrame
    """
    if filename is None:
        # Default to shared saved_data location
        if run_dir is None:
            from config.SNOW_config import MAIN_RUNS_DIR
            saved_data_dir = os.path.join(MAIN_RUNS_DIR, "saved_data")
        else:
            saved_data_dir = get_saved_data_dir(run_dir)

        filename = os.path.join(saved_data_dir, "dataframe_latest.csv")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No saved dataframe found at {filename}")

    df = pd.read_csv(filename)
    print(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns from {filename}")
    return df


def save_df(df, run_dir=None):
    """
    Save dataframe to both timestamped and latest CSV files.

    Args:
        df: pandas DataFrame to save
        run_dir: Run directory (optional, for workflow runs)
    """
    saved_data_dir = get_saved_data_dir(run_dir)

    # Create directory if needed
    os.makedirs(saved_data_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save dataframe to CSV file with timestamp
    timestamped_filename = os.path.join(saved_data_dir, f"dataframe_{timestamp}.csv")
    df.to_csv(timestamped_filename, index=False)
    print(f"Saved dataframe to {timestamped_filename}")

    # Also save to a fixed filename for easy access to the latest version
    latest_filename = os.path.join(saved_data_dir, "dataframe_latest.csv")
    df.to_csv(latest_filename, index=False)
    print(f"Also saved as {latest_filename}")
