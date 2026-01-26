from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .pytorch_autoencoder import autoencoder_scores

def add_anomaly_scores(
    X_train: pd.DataFrame | np.ndarray,
    X_val: Optional[pd.DataFrame | np.ndarray] = None,
    X_test: Optional[pd.DataFrame | np.ndarray] = None,
    method: str = "IsolationForest",
    random_state: int = 42,
    contamination: float = 0.01,
    ae_hidden_dim: int = 128,
    ae_latent_dim: int = 32,
    ae_epochs: int = 30,
    ae_batch_size: int = 256,
    ae_lr: float = 1e-3,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fit an anomaly detector on X_train and append anomaly scores as a feature.
    """
    if method == "None":
        X_train_df = _ensure_df(X_train)
        X_val_df = _ensure_df(X_val) if X_val is not None else None
        X_test_df = _ensure_df(X_test) if X_test is not None else None
        return X_train_df, X_val_df, X_test_df

    X_train_df = _ensure_df(X_train).copy()
    X_val_df = _ensure_df(X_val).copy() if X_val is not None else None
    X_test_df = _ensure_df(X_test).copy() if X_test is not None else None

    if method == "IsolationForest":
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
    elif method == "LOF":
        from sklearn.neighbors import LocalOutlierFactor

        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,
        )
    elif method == "Autoencoder":
        score_col = "anomaly_score_autoencoder"
        train_scores, val_scores, test_scores = autoencoder_scores(
            X_train_df,
            X_val_df,
            X_test_df,
            hidden_dim=ae_hidden_dim,
            latent_dim=ae_latent_dim,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            lr=ae_lr,
            random_state=random_state,
        )
        X_train_df[score_col] = train_scores
        if X_val_df is not None and val_scores is not None:
            X_val_df[score_col] = val_scores
        if X_test_df is not None and test_scores is not None:
            X_test_df[score_col] = test_scores
        return X_train_df, X_val_df, X_test_df
    else:
        raise ValueError(f"Unknown anomaly method: {method}")

    model.fit(X_train_df)

    score_col = f"anomaly_score_{method.lower()}"
    X_train_df[score_col] = model.decision_function(X_train_df)

    if X_val_df is not None:
        X_val_df[score_col] = model.decision_function(X_val_df)
    if X_test_df is not None:
        X_test_df[score_col] = model.decision_function(X_test_df)

    return X_train_df, X_val_df, X_test_df


def _ensure_df(data: Optional[pd.DataFrame | np.ndarray]) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])

