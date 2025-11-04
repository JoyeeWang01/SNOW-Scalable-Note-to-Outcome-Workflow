"""
Custom machine learning models for evaluation.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import RidgeClassifier


class RandomFeatureModel(BaseEstimator, ClassifierMixin):
    """
    Random Feature Model using random projections with ReLU activation.

    This model projects input features onto random unit vectors, applies ReLU
    activation, and trains a ridge classifier on the transformed features.

    Parameters:
        n_components: Number of random features to generate
        alpha: Regularization strength for ridge classifier
        random_state: Random seed for reproducibility
    """

    def __init__(self, n_components: int = 100, alpha: float = 1.0, random_state: int = None):
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the random feature model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            self
        """
        # Set random seed
        rng = np.random.RandomState(self.random_state)

        # Store number of features
        n_samples, n_features = X.shape

        # Generate random Gaussian vectors
        self.random_weights_ = rng.randn(n_features, self.n_components)

        # Normalize to unit vectors
        self.random_weights_ = self.random_weights_ / np.linalg.norm(
            self.random_weights_, axis=0, keepdims=True
        )

        # Transform features: apply random projection and ReLU
        X_transformed = np.maximum(0, X @ self.random_weights_)

        # Train ridge classifier on transformed features
        self.ridge_ = RidgeClassifier(alpha=self.alpha, random_state=self.random_state)
        self.ridge_.fit(X_transformed, y)

        # Store classes
        self.classes_ = self.ridge_.classes_

        return self

    def _transform(self, X):
        """Apply random projection and ReLU activation."""
        return np.maximum(0, X @ self.random_weights_)

    def decision_function(self, X):
        """
        Compute decision function.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Decision values (n_samples,)
        """
        X_transformed = self._transform(X)
        return self.ridge_.decision_function(X_transformed)

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        X_transformed = self._transform(X)
        return self.ridge_.predict(X_transformed)

    def predict_proba(self, X):
        """
        Predict class probabilities using sigmoid transformation.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability estimates (n_samples, n_classes)
        """
        from scipy.special import expit  # sigmoid function

        decision = self.decision_function(X)

        # Apply sigmoid to get probabilities
        if len(self.classes_) == 2:
            # Binary classification
            prob_positive = expit(decision)
            prob_negative = 1 - prob_positive
            return np.vstack([prob_negative, prob_positive]).T
        else:
            # Multi-class (not typically used but included for completeness)
            probs = expit(decision)
            return probs / probs.sum(axis=1, keepdims=True)


class SVDImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values using Singular Value Decomposition (SVD).

    This imputer iteratively reconstructs the data matrix using truncated SVD,
    updating only the missing values until convergence.

    Parameters:
        n_components: Number of SVD components to keep
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: int = None
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the SVD imputer (stores column means for initial imputation).

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        # Convert DataFrame to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values

        # Check if X contains non-numeric data
        if X.dtype == np.object_ or X.dtype.kind not in ('f', 'i', 'u'):
            raise TypeError(
                f"SVDImputer received non-numeric data with dtype {X.dtype}. "
                f"All features must be numeric. Please check that categorical or "
                f"string columns (like MRN, patient IDs) have been removed from the data."
            )

        # Store column means for initial imputation
        self.column_means_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        """
        Impute missing values using iterative SVD reconstruction.

        Args:
            X: Feature matrix with missing values (n_samples, n_features)

        Returns:
            Imputed feature matrix (n_samples, n_features)
        """
        # Convert DataFrame to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values

        # Check if X contains non-numeric data
        if X.dtype == np.object_ or X.dtype.kind not in ('f', 'i', 'u'):
            raise TypeError(
                f"SVDImputer received non-numeric data with dtype {X.dtype}. "
                f"All features must be numeric. Please check that categorical or "
                f"string columns (like MRN, patient IDs) have been removed from the data."
            )

        # Make a copy to avoid modifying original
        X_imputed = X.copy()

        # Track which values are missing
        missing_mask = np.isnan(X)

        # Initial imputation with column means
        for col_idx in range(X.shape[1]):
            mask = missing_mask[:, col_idx]
            if mask.any():
                X_imputed[mask, col_idx] = self.column_means_[col_idx]

        # Iterative SVD reconstruction
        for iteration in range(self.max_iter):
            # Store previous values for convergence check
            X_prev = X_imputed.copy()

            # Perform SVD (truncated to n_components)
            try:
                U, s, Vt = np.linalg.svd(X_imputed, full_matrices=False)

                # Truncate to n_components
                k = min(self.n_components, len(s))
                U_k = U[:, :k]
                s_k = s[:k]
                Vt_k = Vt[:k, :]

                # Reconstruct matrix
                X_reconstructed = U_k @ np.diag(s_k) @ Vt_k

                # Update only missing values
                X_imputed[missing_mask] = X_reconstructed[missing_mask]

            except np.linalg.LinAlgError:
                # SVD failed to converge, return current imputation
                break

            # Check convergence
            diff = np.linalg.norm(X_imputed[missing_mask] - X_prev[missing_mask])
            if diff < self.tol:
                break

        return X_imputed
