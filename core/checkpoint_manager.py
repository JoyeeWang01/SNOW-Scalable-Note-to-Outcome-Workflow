"""
Checkpoint management for feature processing.

This module provides reusable checkpoint functions for:
- Feature definition workflow (chunk-based processing)
- Feature extraction and validation workflow (loop-based processing)
"""

import os
import json
import shutil
from datetime import datetime
from core.log_utils import print


def load_aligned_feature_chunks(checkpoint_dir):
    """
    Load list of completed chunks from checkpoint file for feature alignment.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        List of completed chunk indices
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    completed_chunks_file = os.path.join(checkpoint_dir, "completed_chunks.json")

    completed_chunks = []
    if os.path.exists(completed_chunks_file):
        with open(completed_chunks_file, 'r') as f:
            completed_data = json.load(f)
            completed_chunks = completed_data.get('completed_chunks', [])
        print(f"Found {len(completed_chunks)} completed chunks from previous run")

    return completed_chunks


def save_align_feature_progress(checkpoint_dir, total_chunks, current_chunk, completed_chunks):
    """
    Save overall progress information for feature alignment workflow.

    Args:
        checkpoint_dir: Directory for checkpoint files
        total_chunks: Total number of chunks
        current_chunk: Current chunk index
        completed_chunks: List of completed chunk indices
    """
    align_progress_file = os.path.join(checkpoint_dir, "align_feature_progress.json")
    align_progress_data = {
        'total_chunks': total_chunks,
        'current_chunk': current_chunk,
        'completed_chunks': completed_chunks,
        'timestamp': datetime.now().isoformat()
    }
    with open(align_progress_file, 'w') as f:
        json.dump(align_progress_data, f, indent=2)


def mark_chunk_completed(checkpoint_dir, chunk_idx, completed_chunks, clean_features):
    """
    Mark a chunk as completed and save its features.

    Args:
        checkpoint_dir: Directory for checkpoint files
        chunk_idx: Index of the completed chunk
        completed_chunks: List of completed chunk indices (will be modified)
        clean_features: Cleaned features for this chunk

    Returns:
        Updated completed_chunks list
    """
    # Save cleaned features for this chunk
    chunk_features_file = os.path.join(checkpoint_dir, f"completed_chunk_{chunk_idx}.json")
    with open(chunk_features_file, 'w') as f:
        json.dump(clean_features, f, indent=2)

    # Update completed chunks list
    completed_chunks.append(chunk_idx)
    completed_data = {
        'completed_chunks': completed_chunks,
        'timestamp': datetime.now().isoformat()
    }
    completed_chunks_file = os.path.join(checkpoint_dir, "completed_chunks.json")
    with open(completed_chunks_file, 'w') as f:
        json.dump(completed_data, f, indent=2)

    return completed_chunks


def cleanup_checkpoints(checkpoint_dir, completed_chunks, total_chunks):
    """
    Clean up checkpoint files after all chunks are completed.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        completed_chunks: List of completed chunk indices
        total_chunks: Total number of chunks
    """
    if len(completed_chunks) == total_chunks:
        print("Cleaning up checkpoint files...")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print("Checkpoint files cleaned up")


# ============================================================================
# General-purpose checkpoint functions for extraction/validation workflow
# ============================================================================

def save_main_checkpoint(checkpoint_dir, checkpoint_data, filename="main_loop_checkpoint.json"):
    """
    Save main checkpoint with arbitrary data.

    Args:
        checkpoint_dir: Directory for checkpoint files
        checkpoint_data: Dictionary containing checkpoint data
        filename: Name of checkpoint file (default: main_loop_checkpoint.json)

    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, filename)

    # Add timestamp if not present
    if 'timestamp' not in checkpoint_data:
        checkpoint_data['timestamp'] = datetime.now().isoformat()

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    return checkpoint_file


def load_main_checkpoint(checkpoint_dir, filename="main_loop_checkpoint.json"):
    """
    Load main checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        filename: Name of checkpoint file (default: main_loop_checkpoint.json)

    Returns:
        Dictionary containing checkpoint data, or None if checkpoint doesn't exist
    """
    checkpoint_file = os.path.join(checkpoint_dir, filename)

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Loaded checkpoint from {checkpoint_file}")
        return checkpoint_data

    return None


def save_dataframe_checkpoint(checkpoint_dir, df, filename="final_df_checkpoint.csv"):
    """
    Save DataFrame checkpoint.

    Args:
        checkpoint_dir: Directory for checkpoint files
        df: pandas DataFrame to save
        filename: Name of checkpoint file (default: final_df_checkpoint.csv)

    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, filename)
    df.to_csv(checkpoint_file, index=False)
    return checkpoint_file


def load_dataframe_checkpoint(checkpoint_dir, filename="final_df_checkpoint.csv"):
    """
    Load DataFrame checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        filename: Name of checkpoint file (default: final_df_checkpoint.csv)

    Returns:
        pandas DataFrame if checkpoint exists, None otherwise
    """
    import pandas as pd
    checkpoint_file = os.path.join(checkpoint_dir, filename)

    if os.path.exists(checkpoint_file):
        df = pd.read_csv(checkpoint_file)
        print(f"Loaded DataFrame checkpoint with {len(df.columns)} columns from {checkpoint_file}")
        return df

    return None


def save_progress_checkpoint(checkpoint_dir, progress_data, filename):
    """
    Save progress checkpoint for extraction/validation/reextraction.

    Args:
        checkpoint_dir: Directory for checkpoint files
        progress_data: Dictionary containing progress data
        filename: Name of checkpoint file (e.g., "extraction_progress.json")

    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, filename)

    # Add timestamp if not present
    if 'timestamp' not in progress_data:
        progress_data['timestamp'] = datetime.now().isoformat()

    with open(checkpoint_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

    return checkpoint_file


def load_progress_checkpoint(checkpoint_dir, filename):
    """
    Load progress checkpoint for extraction/validation/reextraction.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        filename: Name of checkpoint file (e.g., "extraction_progress.json")

    Returns:
        Dictionary containing progress data, or None if checkpoint doesn't exist
    """
    checkpoint_file = os.path.join(checkpoint_dir, filename)

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            progress_data = json.load(f)
        print(f"Loaded progress checkpoint from {checkpoint_file}")
        return progress_data

    return None


def cleanup_progress_checkpoint(checkpoint_dir, filename):
    """
    Remove a specific progress checkpoint file.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        filename: Name of checkpoint file to remove
    """
    checkpoint_file = os.path.join(checkpoint_dir, filename)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Cleaned up {filename}")


def cleanup_all_checkpoints(checkpoint_dir):
    """
    Remove all checkpoint files and directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files
    """
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Cleaned up all checkpoints in {checkpoint_dir}")


def check_checkpoint_exists(checkpoint_dir, filename):
    """
    Check if a checkpoint file exists.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        filename: Name of checkpoint file

    Returns:
        Boolean indicating if checkpoint exists
    """
    checkpoint_file = os.path.join(checkpoint_dir, filename)
    return os.path.exists(checkpoint_file)
