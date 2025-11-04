"""
Generate Gemini embeddings from clinical notes using Vertex AI API.

This script:
1. Loads clinical notes from configured data file
2. Generates embeddings using Gemini embedding model via Vertex AI
3. Handles batching for efficient API usage
4. Saves embeddings as CSV

Based on: mimic/gemini_embedding.ipynb
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

from config.SNOW_config import NOTES_FILE_PATH, NOTES_COL
from config.api_gemini import PROJECT_ID, LOCATION


def preprocess_clinical_text(text: str) -> str:
    """
    Preprocess clinical text for LLM embedding models.

    For neural embedding models like Gemini, we apply minimal preprocessing:
    - Clean formatting issues (extra whitespace, special characters)
    - Preserve medical terminology, numbers, and clinical context
    - Keep natural language structure for better contextual embeddings

    Args:
        text: Raw clinical note text

    Returns:
        Preprocessed text suitable for LLM embeddings
    """
    # Convert to string and strip
    text = str(text).strip()

    # Remove URLs (not clinically relevant)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Normalize excessive whitespace (including newlines, tabs)
    text = re.sub(r'\s+', ' ', text)

    # Remove excessive punctuation (e.g., "!!!" -> "!")
    text = re.sub(r'([.!?])\1+', r'\1', text)

    # Final cleanup
    text = text.strip()

    return text


def initialize_vertex_ai():
    """
    Initialize Vertex AI with configured project and location.

    Returns:
        tuple: (project_id, location)
    """
    try:
        print(f"Using configured project: {PROJECT_ID}")
        print(f"Using configured location: {LOCATION}")

        # Initialize Vertex AI SDK with configured values
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print(f"✅ Vertex AI initialized")

        return PROJECT_ID, LOCATION

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Vertex AI: {e}")


def get_gemini_embedding(text: str, model: TextEmbeddingModel, max_retries: int = 3) -> list[float]:
    """
    Get embedding for a single text using Gemini model.

    Args:
        text: Text to embed
        model: Gemini embedding model instance
        max_retries: Maximum number of retry attempts

    Returns:
        List of embedding values
    """
    for attempt in range(max_retries):
        try:
            embeddings = model.get_embeddings([text])
            for embedding in embeddings:
                return embedding.values
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s (error: {e})")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                raise


def generate_embeddings(
    texts: list[str],
    model: TextEmbeddingModel,
    progress_save_freq: int = 10,
    checkpoint_file: str = "data/embeddings/gemini_embeddings_checkpoint.csv"
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts with checkpointing.

    Args:
        texts: List of texts to embed
        model: Gemini embedding model instance
        progress_save_freq: Save progress every N texts
        checkpoint_file: File to save progress checkpoints

    Returns:
        List of embedding vectors
    """
    embeddings = []
    total = len(texts)

    # Check for existing checkpoint
    start_idx = 0
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file: {checkpoint_file}")
        checkpoint_df = pd.read_csv(checkpoint_file, header=None)
        embeddings = checkpoint_df.values.tolist()
        start_idx = len(embeddings)
        print(f"Resuming from index {start_idx}/{total}")

    if start_idx >= total:
        print("All embeddings already generated!")
        return embeddings

    print(f"\nGenerating embeddings for {total} texts...")
    print(f"Progress checkpoint every {progress_save_freq} texts")

    for i in range(start_idx, total):
        text = texts[i]
        try:
            # Get embedding for single text
            embedding = get_gemini_embedding(text, model)
            embeddings.append(embedding)

            # Save checkpoint and report progress
            if (i + 1) % progress_save_freq == 0 or (i + 1) == total:
                # Save checkpoint
                checkpoint_df = pd.DataFrame(embeddings)
                os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
                checkpoint_df.to_csv(checkpoint_file, index=False, header=False)
                print(f"Progress: {i + 1}/{total} ({100*(i+1)/total:.1f}%) - Checkpoint saved")

        except Exception as e:
            print(f"Error at index {i}: {e}")
            # Append zero vector as placeholder
            print(f"  Using zero vector as placeholder")
            if embeddings:
                embeddings.append([0.0] * len(embeddings[0]))
            else:
                embeddings.append([0.0] * 768)  # Default Gemini embedding dimension

    return embeddings


def main():
    """Generate Gemini embeddings."""
    print("="*80)
    print("GEMINI EMBEDDING GENERATION (VERTEX AI)")
    print("="*80)

    # Initialize Vertex AI
    print("\nInitializing Vertex AI...")
    project_id, location = initialize_vertex_ai()

    # Load embedding model
    print("\nLoading Gemini embedding model...")
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    print("Model loaded: gemini-embedding-001")

    # Load data
    print(f"\nLoading data from: {NOTES_FILE_PATH}")
    data = pd.read_csv(NOTES_FILE_PATH)
    print(f"Loaded {len(data)} records")

    # Extract and preprocess texts
    print("\nPreprocessing clinical texts...")
    raw_texts = data[NOTES_COL].tolist()
    texts = [preprocess_clinical_text(text) for text in raw_texts]
    print(f"Preprocessed {len(texts)} texts")

    # Generate embeddings with checkpointing
    checkpoint_file = "data/embeddings/gemini_embeddings_checkpoint.csv"
    embeddings = generate_embeddings(
        texts=texts,
        model=model,
        progress_save_freq=50,  # Save checkpoint every 50 embeddings
        checkpoint_file=checkpoint_file
    )

    # Convert to DataFrame
    embeddings_df = pd.DataFrame(embeddings)
    print(f"\nEmbedding matrix shape: {embeddings_df.shape}")

    # Save embeddings
    output_dir = "data/embeddings"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Embedding shape: {embeddings_df.shape} (rows x columns)")

    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Checkpoint file removed: {checkpoint_file}")

    # Perform SVD only if embedding dimensions > number of notes
    n_rows, n_dims = embeddings_df.shape
    if n_dims > n_rows:
        print(f"\nPerforming SVD (dimensions {n_dims} > samples {n_rows})...")
        u, _, _ = np.linalg.svd(embeddings_df.values)

        # Use top components (up to number of samples)
        n_components = n_rows
        embeddings_svd = u[:, 0:n_components]

        print(f"SVD reduced shape: {embeddings_svd.shape}")

        # Save SVD embeddings
        svd_output_file = os.path.join(output_dir, "gemini_embeddings.csv")
        pd.DataFrame(embeddings_svd).to_csv(svd_output_file, index=False, header=False)
        print(f"SVD embeddings saved to: {svd_output_file}")
        print(f"SVD file shape: {embeddings_svd.shape} (rows x columns)")
    else:
        print(f"\nSkipping SVD (dimensions {n_dims} <= samples {n_rows})")

        output_file = os.path.join(output_dir, "gemini_embeddings.csv")
        embeddings_df.to_csv(output_file, index=False, header=False)
        print(f"\nEmbeddings saved to: {output_file}")
        print(f"File shape: {embeddings_df.shape} (rows x columns)")

    print("\n" + "="*80)
    print("GEMINI EMBEDDING GENERATION COMPLETE")
    print("="*80)
    print(f"\nProject charged: {project_id}")
    print(f"Total API calls: {len(texts)}")


if __name__ == "__main__":
    main()
