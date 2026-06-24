"""
unsupervised.py — Anomaly detection for the no-label scenario.

Provides Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic
Envelope, and an optional Keras autoencoder. Raw scores are turned into a
human-readable 0–100 risk score and per-row plain-English explanations of *why*
a transaction is unusual (largest standardized feature deviations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from .constants import RANDOM_STATE


def available_detectors() -> list[str]:
    base = ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Elliptic Envelope"]
    try:
        import tensorflow  # noqa: F401
        base.append("Autoencoder")
    except Exception:
        pass
    return base


def _raw_scores(name: str, X: np.ndarray, contamination: float):
    """Return a per-row anomaly score where HIGHER == more anomalous."""
    if name == "Isolation Forest":
        m = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1).fit(X)
        return -m.decision_function(X)
    if name == "Local Outlier Factor":
        m = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        m.fit_predict(X)
        return -m.negative_outlier_factor_
    if name == "One-Class SVM":
        m = OneClassSVM(nu=min(max(contamination, 0.01), 0.5), gamma="scale").fit(X)
        return -m.decision_function(X)
    if name == "Elliptic Envelope":
        m = EllipticEnvelope(contamination=contamination, random_state=RANDOM_STATE,
                             support_fraction=1.0).fit(X)
        return -m.decision_function(X)
    if name == "Autoencoder":
        return _autoencoder_scores(X)
    raise ValueError(f"Unknown detector '{name}'.")


def _autoencoder_scores(X: np.ndarray) -> np.ndarray:
    """Reconstruction error from a small dense autoencoder. Falls back to
    Isolation Forest if TensorFlow is unavailable."""
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_STATE)
        n_in = X.shape[1]
        enc = max(2, n_in // 2)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_in,)),
            tf.keras.layers.Dense(enc, activation="relu"),
            tf.keras.layers.Dense(max(2, enc // 2), activation="relu"),
            tf.keras.layers.Dense(enc, activation="relu"),
            tf.keras.layers.Dense(n_in, activation="linear"),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, X, epochs=20, batch_size=64, verbose=0, validation_split=0.1)
        recon = model.predict(X, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)
    except Exception:
        m = IsolationForest(random_state=RANDOM_STATE).fit(X)
        return -m.decision_function(X)


def risk_score(scores: np.ndarray) -> np.ndarray:
    """Map raw anomaly scores onto an interpretable 0–100 risk scale via
    percentile ranking (robust to the score's arbitrary units)."""
    s = pd.Series(scores)
    return (s.rank(pct=True) * 100).to_numpy()


def smart_threshold(scores: np.ndarray, contamination: float) -> float:
    """Auto-pick the anomaly cutoff at the (1 - contamination) percentile."""
    return float(np.quantile(scores, 1 - contamination))


def run_detector(name: str, X: np.ndarray, contamination: float = 0.05) -> dict:
    scores = _raw_scores(name, X, contamination)
    risk = risk_score(scores)
    thr = smart_threshold(scores, contamination)
    flagged = (scores >= thr).astype(int)
    return {"name": name, "scores": scores, "risk": risk,
            "threshold": thr, "flagged": flagged,
            "n_flagged": int(flagged.sum()), "contamination": contamination}


def explain_anomaly(X_scaled_row: np.ndarray, feature_cols: list[str], top_k: int = 5) -> list[dict]:
    """Explain a single (already standardized) row: the features that deviate
    most from normal (|z| largest) are what make it anomalous."""
    row = np.asarray(X_scaled_row).ravel()
    order = np.argsort(-np.abs(row))[:top_k]
    out = []
    for idx in order:
        z = float(row[idx])
        direction = "unusually high" if z > 0 else "unusually low"
        out.append({"feature": feature_cols[idx], "z": round(z, 2),
                    "direction": direction, "severity": min(abs(z) / 3, 1.0)})
    return out


def anomaly_sentence(explanations: list[dict]) -> str:
    if not explanations:
        return "No single feature stands out — this row is broadly typical."
    parts = [f"**{e['feature']}** is {e['direction']} (z={e['z']})" for e in explanations[:3]]
    return "Flagged because " + "; ".join(parts) + "."
