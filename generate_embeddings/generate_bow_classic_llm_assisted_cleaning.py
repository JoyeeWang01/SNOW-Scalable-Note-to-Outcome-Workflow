"""
Generate classic Bag-of-Words (BoW) embeddings from clinical notes.

This script:
1. Loads clinical notes from configured data file
2. Preprocesses text (lowercase, remove punctuation, lemmatization, stopword removal)
3. Generates classic BoW vectors using CountVectorizer
4. Performs SVD to reduce dimensionality
5. Saves the embeddings as CSV

Adapted from: https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import nltk
from sklearn import feature_extraction

from config.SNOW_config import NOTES_FILE_PATH, NOTES_COL


def get_clinical_stopwords():
    """
    Get stopwords for clinical text, excluding clinically important words.

    Returns:
        Set of stopwords safe to remove from clinical notes
    """
    # Start with standard English stopwords
    base_stopwords = set(nltk.corpus.stopwords.words("english"))

    # Clinical terms that should NOT be removed (even if in standard stopwords)
    clinical_keep_words = {
        'no', 'not', 'nor', 'without', 'against',  # Negations
        'positive', 'negative',  # Test results
        'above', 'below', 'over', 'under',  # Measurements/comparisons
        'few', 'more', 'most', 'all', 'any',  # Quantifiers
        'same', 'other', 'both', 'each',  # Clinical comparisons
    }

    # Remove clinical terms from stopwords
    clinical_stopwords = base_stopwords - clinical_keep_words

    return clinical_stopwords


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    """
    Preprocess clinical text with domain-specific handling.

    Args:
        text: string - text to preprocess
        flg_stemm: bool - whether stemming is to be applied
        flg_lemm: bool - whether lemmatization is to be applied
        lst_stopwords: list - list of stopwords to remove

    Returns:
        Cleaned text
    """
    # Convert to string and lowercase
    text = str(text).lower().strip()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Normalize measurement units (preserve numbers + units together)
    # e.g., "3.5cm" -> "3.5 cm", "10mg" -> "10 mg"
    text = re.sub(r'(\d+\.?\d*)([a-z]{1,3})\b', r'\1 \2', text)

    # Remove punctuation but keep alphanumeric (preserves T2, N0, M0, etc.)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize (convert from string to list)
    lst_text = text.split()

    # Keep all tokens including short ones (medical abbreviations like T2, ER, PR)
    # No length filtering for clinical text

    # Remove stopwords (using clinical-specific list)
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Stemming (remove -ing, -ly, ...)
    if flg_stemm:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    # Lemmatisation (convert the word into root word)
    if flg_lemm:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # Back to string from list
    text = " ".join(lst_text)
    return text


def main():
    """Generate classic BoW embeddings."""
    print("="*80)
    print("CLASSIC BAG-OF-WORDS EMBEDDING GENERATION")
    print("="*80)

    # Download required NLTK data
    print("\nDownloading NLTK resources...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK resources: {e}")

    # Load clinical stopwords
    print("Loading clinical-specific stopwords...")
    lst_stopwords = get_clinical_stopwords()
    print(f"Loaded {len(lst_stopwords)} stopwords (excluding clinically important terms)")

    # Load data
    print(f"\nLoading data from: {NOTES_FILE_PATH}")
    data = pd.read_csv(NOTES_FILE_PATH)
    print(f"Loaded {len(data)} records")

    # Create dataframe
    dtf = data.copy()
    dtf.rename(columns={NOTES_COL: "text"}, inplace=True)

    # Preprocess text
    print("\nPreprocessing text...")
    dtf["text_clean"] = dtf["text"].apply(
        lambda x: utils_preprocess_text(
            x,
            flg_stemm=False,
            flg_lemm=True,
            lst_stopwords=lst_stopwords
        )
    )
    print("Text preprocessing complete")

    # Generate classic BoW vectors
    print("\nGenerating classic BoW vectors...")
    vectorizer_classic = feature_extraction.text.CountVectorizer(
        max_features=10000,
        ngram_range=(1, 5)
    )

    corpus = dtf['text_clean']
    vectorizer_classic.fit(corpus)
    X_classic_bow = vectorizer_classic.transform(corpus)

    print(f"BoW matrix shape: {X_classic_bow.shape}")
    print(f"Number of features: {len(vectorizer_classic.vocabulary_)}")

    # Perform SVD
    print("\nPerforming SVD...")
    u_classic, s_classic, vh_classic = np.linalg.svd(X_classic_bow.todense())

    # Use top components (up to number of samples)
    n_components = min(len(data), X_classic_bow.shape[1])
    embeddings = u_classic[:, 0:n_components]

    print(f"Embedding shape: {embeddings.shape}")

    # Save embeddings
    output_dir = "data/embeddings"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "bow_classic_llm_cleaned.csv")

    pd.DataFrame(embeddings).to_csv(output_file, index=False, header=False)
    print(f"\nEmbeddings saved to: {output_file}")
    print(f"File shape: {embeddings.shape} (rows x columns)")

    print("\n" + "="*80)
    print("CLASSIC BOW EMBEDDING GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
