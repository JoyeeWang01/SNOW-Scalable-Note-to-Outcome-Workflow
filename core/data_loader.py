"""
Data loading and preprocessing functions for evaluation.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from core.log_utils import print


def load_structured_features(
    structured_path: str,
    label_col: str,
    index_col: Optional[str] = None,
    notes_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load structured features and labels.

    Args:
        structured_path: Path to structured features file
        label_col: Name of label column
        index_col: Index column to remove (optional)
        notes_col: Notes column to remove (optional)

    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    df = pd.read_csv(structured_path)

    # Extract labels
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {structured_path}")

    y = df[label_col]

    # Remove label and non-feature columns
    # Typically remove: unnamed index columns, identifiers, label, notes
    drop_cols = [col for col in df.columns if col.startswith('Unnamed')]

    # Always remove label column
    drop_cols.append(label_col)
    print(f"    Removing LABEL_COL '{label_col}' from structured features")

    # Remove index column if specified
    if index_col is not None and index_col in df.columns:
        drop_cols.append(index_col)
        print(f"    Removing INDEX_COL '{index_col}' from structured features")

    # Remove notes column if specified
    if notes_col is not None and notes_col in df.columns:
        drop_cols.append(notes_col)
        print(f"    Removing NOTES_COL '{notes_col}' from structured features")

    X = df.drop(columns=drop_cols, errors='ignore')

    return X, y


def load_snow_features(snow_path: str, index_col: Optional[str] = None, label_col: Optional[str] = None, notes_col: Optional[str] = None) -> pd.DataFrame:
    """
    Load SNOW-extracted features.

    Args:
        snow_path: Path to SNOW features CSV
        index_col: Index column to remove (optional)
        label_col: Label column to remove (optional)
        notes_col: Notes column to remove (optional)

    Returns:
        DataFrame with SNOW features
    """
    if not os.path.exists(snow_path):
        raise FileNotFoundError(f"SNOW features file not found: {snow_path}")

    df = pd.read_csv(snow_path)

    # Remove unnamed index columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # Remove index column if specified
    if index_col is not None and index_col in df.columns:
        print(f"    Removing INDEX_COL '{index_col}' from SNOW features")
        df = df.drop(columns=[index_col])

    # Remove label column if specified
    if label_col is not None and label_col in df.columns:
        print(f"    Removing LABEL_COL '{label_col}' from SNOW features")
        df = df.drop(columns=[label_col])

    # Remove notes column if specified
    if notes_col is not None and notes_col in df.columns:
        print(f"    Removing NOTES_COL '{notes_col}' from SNOW features")
        df = df.drop(columns=[notes_col])

    return df


def load_embeddings(embedding_path: str) -> pd.DataFrame:
    """
    Load embedding features (BoW or Gemini).

    Args:
        embedding_path: Path to embedding CSV

    Returns:
        DataFrame with embedding features
    """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    # Embeddings are saved without headers
    df = pd.read_csv(embedding_path, header=None)

    # Add column names
    df.columns = [f"emb_{i}" for i in range(df.shape[1])]

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess feature DataFrame.

    - Remove high-cardinality categorical columns (>10% of rows are unique categories)
    - One-hot encode categorical variables
    - Remove columns with >95% missing values
    - Handle infinite values

    Args:
        df: Feature DataFrame

    Returns:
        Preprocessed DataFrame
    """
    # Identify categorical columns (object or category dtype)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Remove high-cardinality categorical columns
    # If unique values > 10% of data, it's likely free text or high-cardinality identifiers
    high_cardinality_threshold = 0.10
    cols_to_remove = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        cardinality_ratio = n_unique / len(df)
        if cardinality_ratio > high_cardinality_threshold:
            cols_to_remove.append(col)
            print(f"  Removing high-cardinality column '{col}': {n_unique} unique values ({cardinality_ratio:.1%} of data)")

    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        # Update cat_cols to exclude removed columns
        cat_cols = [col for col in cat_cols if col not in cols_to_remove]

    # One-hot encode remaining categoricals
    if len(cat_cols) > 0:
        print(f"  One-hot encoding {len(cat_cols)} categorical columns: {list(cat_cols)}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    # Remove columns with >95% missing
    missing_threshold = 0.95
    missing_pct = df.isnull().sum() / len(df)
    cols_to_remove_missing = missing_pct[missing_pct >= missing_threshold].index.tolist()

    if cols_to_remove_missing:
        for col in cols_to_remove_missing:
            print(f"  Removing column with high missing values '{col}': {missing_pct[col]:.1%} missing")
        df = df.drop(columns=cols_to_remove_missing)

    # Replace infinities with NaN (will be imputed later)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Final check: ensure all columns are numeric
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"  ⚠️  WARNING: After preprocessing, found non-numeric columns: {non_numeric_cols}")
        for col in non_numeric_cols:
            print(f"    Column '{col}' has dtype: {df[col].dtype}, unique values: {df[col].unique()[:10]}")
        print(f"    Converting to numeric or removing...")

        # Try to convert to numeric, or remove if conversion fails
        for col in non_numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"    Successfully converted '{col}' to numeric")
            except Exception as e:
                print(f"    Failed to convert '{col}' to numeric, removing it: {e}")
                df = df.drop(columns=[col])

    return df


def combine_feature_sets(
    sources: List[str],
    structured_df: pd.DataFrame,
    snow_df: pd.DataFrame = None,
    embeddings_dfs: Dict[str, pd.DataFrame] = None,
    additional_dfs: Dict[str, pd.DataFrame] = None,
    # Deprecated parameters for backward compatibility
    bow_classic_df: pd.DataFrame = None,
    bow_tfidf_df: pd.DataFrame = None,
    gemini_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Combine multiple feature sets horizontally.

    Args:
        sources: List of source names to include
        structured_df: Structured features DataFrame
        snow_df: SNOW features DataFrame (optional)
        embeddings_dfs: Dictionary of embedding DataFrames (optional)
                       Keys should match embedding names from EMBEDDING_FILES
        additional_dfs: Dictionary of additional feature DataFrames (optional)
        bow_classic_df: DEPRECATED - use embeddings_dfs instead
        bow_tfidf_df: DEPRECATED - use embeddings_dfs instead
        gemini_df: DEPRECATED - use embeddings_dfs instead

    Returns:
        Combined feature DataFrame
    """
    dfs_to_concat = []

    # Merge deprecated parameters into embeddings_dfs for backward compatibility
    if embeddings_dfs is None:
        embeddings_dfs = {}
    else:
        embeddings_dfs = embeddings_dfs.copy()  # Don't modify original

    # Add deprecated parameters if provided
    if bow_classic_df is not None:
        embeddings_dfs['bow_classic'] = bow_classic_df
    if bow_tfidf_df is not None:
        embeddings_dfs['bow_tfidf'] = bow_tfidf_df
    if gemini_df is not None:
        embeddings_dfs['gemini'] = gemini_df

    for source in sources:
        if source == 'structured':
            dfs_to_concat.append(structured_df)
        elif source == 'snow':
            if snow_df is None:
                raise ValueError("SNOW features requested but not provided")
            dfs_to_concat.append(snow_df)
        elif source in embeddings_dfs:
            # Check if it's an embedding from embeddings_dfs
            dfs_to_concat.append(embeddings_dfs[source])
        elif additional_dfs is not None and source in additional_dfs:
            # Check if it's an additional feature set
            dfs_to_concat.append(additional_dfs[source])
        else:
            raise ValueError(f"Unknown feature source: {source}")

    # Combine horizontally
    if len(dfs_to_concat) == 0:
        raise ValueError("No feature sources specified")

    combined_df = pd.concat(dfs_to_concat, axis=1)

    # Ensure no duplicate column names
    if combined_df.columns.duplicated().any():
        # Add suffixes to duplicates
        cols = pd.Series(combined_df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_idx = cols[cols == dup].index
            for i, idx in enumerate(dup_idx):
                cols.iloc[idx] = f"{dup}_{i}"
        combined_df.columns = cols

    return combined_df


def validate_row_alignment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    index_col: str,
    df1_name: str = "df1",
    df2_name: str = "df2"
) -> bool:
    """
    Validate that two DataFrames have matching row order based on index column.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        index_col: Column name to use for alignment validation
        df1_name: Name of first DataFrame (for error messages)
        df2_name: Name of second DataFrame (for error messages)

    Returns:
        True if row alignment matches

    Raises:
        ValueError: If row alignment doesn't match
    """
    if index_col not in df1.columns:
        raise ValueError(f"Index column '{index_col}' not found in {df1_name}")
    if index_col not in df2.columns:
        raise ValueError(f"Index column '{index_col}' not found in {df2_name}")

    # Check if index values match
    index1 = df1[index_col].tolist()
    index2 = df2[index_col].tolist()

    if len(index1) != len(index2):
        raise ValueError(
            f"Row count mismatch: {df1_name} has {len(index1)} rows, "
            f"{df2_name} has {len(index2)} rows"
        )

    if index1 != index2:
        # Find first mismatch
        for i, (val1, val2) in enumerate(zip(index1, index2)):
            if val1 != val2:
                raise ValueError(
                    f"Row alignment mismatch at position {i}: "
                    f"{df1_name}['{index_col}'] = {val1}, "
                    f"{df2_name}['{index_col}'] = {val2}"
                )

    print(f"✓ Row alignment validated: {df1_name} and {df2_name} match on '{index_col}' ({len(index1)} rows)")
    return True


def load_all_features(
    structured_path: str,
    label_col: str,
    index_col: Optional[str] = None,
    notes_col: Optional[str] = None,
    snow_path: str = None,
    embedding_paths: Dict[str, str] = None,
    additional_paths: Dict[str, str] = None
) -> Dict[str, any]:
    """
    Load all feature sets and labels with optional row alignment validation.

    Args:
        structured_path: Path to structured features file
        label_col: Name of label column
        index_col: Column for row alignment validation (None to skip)
        notes_col: Notes column to remove from all feature files (None to skip)
        snow_path: Path to SNOW features (optional)
        embedding_paths: Dictionary of embedding paths {name: path} (optional)
                        Keys should be: 'bow_classic', 'bow_tfidf', 'gemini', etc.
        additional_paths: Dictionary of additional feature paths {name: path} (optional)

    Returns:
        Dictionary containing:
            - 'structured': Preprocessed structured features
            - 'labels': Target labels
            - 'snow': SNOW features (if provided)
            - 'bow_classic': BoW classic (if provided via embedding_paths)
            - 'bow_tfidf': BoW TF-IDF (if provided via embedding_paths)
            - 'gemini': Gemini embeddings (if provided via embedding_paths)
            - Additional feature sets (if provided via additional_paths)
    """
    print("Loading data...")

    # Load structured features and labels
    print(f"  Loading structured features from: {structured_path}")
    structured_df, y = load_structured_features(structured_path, label_col, index_col, notes_col)

    # Keep index column for validation before preprocessing
    structured_df_raw = pd.read_csv(structured_path)

    print(f"  Preprocessing structured features (before: {structured_df.shape})...")
    structured_df = preprocess_features(structured_df)
    print(f"    Structured features after preprocessing: {structured_df.shape}")
    print(f"    Labels: {y.shape}, positive class: {y.sum()}/{len(y)} ({100*y.mean():.1f}%)")

    if index_col is not None:
        print(f"    Index column '{index_col}' removed from features")

    result = {
        'structured': structured_df,
        'labels': y
    }

    # Load optional feature sets with validation
    if snow_path and os.path.exists(snow_path):
        print(f"  Loading SNOW features from: {snow_path}")

        # Validate row alignment if index_col is provided (before removing index_col)
        if index_col is not None:
            snow_df_raw = pd.read_csv(snow_path)
            validate_row_alignment(
                structured_df_raw, snow_df_raw, index_col,
                "STRUCTURED_FILE_PATH", "SNOW_FEATURES_PATH"
            )

        # Load and remove index_col, label_col, and notes_col
        snow_df = load_snow_features(snow_path, index_col, label_col, notes_col)
        print(f"  Preprocessing SNOW features (before: {snow_df.shape})...")
        snow_df = preprocess_features(snow_df)
        result['snow'] = snow_df
        print(f"    SNOW features after preprocessing: {snow_df.shape}")

    # Load embedding files from dictionary
    if embedding_paths:
        for emb_name, emb_path in embedding_paths.items():
            if emb_path and os.path.exists(emb_path):
                print(f"  Loading {emb_name} from: {emb_path}")
                emb_df = load_embeddings(emb_path)
                result[emb_name] = emb_df
                print(f"    {emb_name}: {emb_df.shape}")

    # Load additional feature sets
    if additional_paths:
        for name, path in additional_paths.items():
            if path and os.path.exists(path):
                print(f"  Loading {name} from: {path}")

                # Load raw dataframe for validation
                add_df_raw = pd.read_csv(path)

                # Validate row alignment if index_col is provided
                if index_col is not None and index_col in add_df_raw.columns:
                    validate_row_alignment(
                        structured_df_raw, add_df_raw, index_col,
                        "STRUCTURED_FILE_PATH", name
                    )

                # Remove index_col, label_col, notes_col if present
                cols_to_drop = []
                if index_col is not None and index_col in add_df_raw.columns:
                    cols_to_drop.append(index_col)
                    print(f"    Removing INDEX_COL '{index_col}' from {name}")
                if label_col is not None and label_col in add_df_raw.columns:
                    cols_to_drop.append(label_col)
                    print(f"    Removing LABEL_COL '{label_col}' from {name}")
                if notes_col is not None and notes_col in add_df_raw.columns:
                    cols_to_drop.append(notes_col)
                    print(f"    Removing NOTES_COL '{notes_col}' from {name}")

                if cols_to_drop:
                    add_df_raw = add_df_raw.drop(columns=cols_to_drop)

                # Remove unnamed columns
                add_df_raw = add_df_raw.loc[:, ~add_df_raw.columns.str.startswith('Unnamed')]

                # Preprocess
                print(f"  Preprocessing {name} (before: {add_df_raw.shape})...")
                add_df = preprocess_features(add_df_raw)
                result[name] = add_df
                print(f"    {name} after preprocessing: {add_df.shape}")

    print("Data loading complete.\n")

    return result
