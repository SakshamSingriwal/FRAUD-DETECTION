"""
model_explainer.py — Explainability: SHAP (optional), permutation-importance
fallback, top risk factors for a single prediction, and plain-English summaries.

Everything degrades gracefully: if SHAP is not installed we fall back to model
feature_importances_ / permutation importance, so the app stays useful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model_trainer import get_proba


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def global_importance(model, X, feature_cols, max_samples=500) -> pd.DataFrame:
    """Global feature importance. Prefers SHAP, then native importances, then
    permutation importance. Returns a sorted (feature, importance) frame."""
    X = np.asarray(X)
    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
        X = X[idx]

    # 1. SHAP (best — works for tree & linear models).
    if shap_available():
        try:
            import shap
            explainer = shap.Explainer(model.predict_proba, X) \
                if not hasattr(model, "get_booster") else shap.TreeExplainer(model)
            vals = explainer(X)
            arr = vals.values
            if arr.ndim == 3:            # (n, features, classes) -> positive class
                arr = arr[:, :, -1]
            imp = np.abs(arr).mean(axis=0)
            return _frame(feature_cols, imp).assign(method="SHAP")
        except Exception:
            pass

    # 2. Native importances.
    if hasattr(model, "feature_importances_"):
        return _frame(feature_cols, np.asarray(model.feature_importances_)).assign(method="native")
    if hasattr(model, "coef_"):
        return _frame(feature_cols, np.abs(np.asarray(model.coef_)).ravel()).assign(method="coefficient")

    # 3. Permutation importance (model-agnostic, last resort).
    try:
        base = get_proba(model, X)
        rng = np.random.default_rng(42)
        scores = []
        for j in range(X.shape[1]):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            scores.append(np.mean(np.abs(get_proba(model, Xp) - base)))
        return _frame(feature_cols, np.asarray(scores)).assign(method="permutation")
    except Exception:
        return _frame(feature_cols, np.ones(len(feature_cols))).assign(method="uniform")


def _frame(cols, imp):
    imp = np.asarray(imp, dtype=float)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return pd.DataFrame({"feature": cols, "importance": imp}).sort_values(
        "importance", ascending=False).reset_index(drop=True)


def shap_values_for(model, X, feature_cols, max_samples=300):
    """Return (shap_array, X_sample) for summary plots, or None if unavailable."""
    if not shap_available():
        return None
    try:
        import shap
        X = np.asarray(X)
        if len(X) > max_samples:
            X = X[np.random.default_rng(42).choice(len(X), max_samples, replace=False)]
        explainer = shap.TreeExplainer(model) if hasattr(model, "get_booster") \
            else shap.Explainer(model.predict_proba, X)
        vals = explainer(X)
        arr = vals.values
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        return arr, X
    except Exception:
        return None


def top_risk_factors(model, x_row_scaled, feature_cols, top_k=5) -> list[dict]:
    """Per-prediction risk factors. SHAP local contribution if available, else a
    perturbation sensitivity proxy."""
    x = np.asarray(x_row_scaled).reshape(1, -1)
    if shap_available():
        try:
            import shap
            explainer = shap.TreeExplainer(model) if hasattr(model, "get_booster") \
                else shap.Explainer(model.predict_proba, x)
            vals = explainer(x)
            arr = vals.values
            if arr.ndim == 3:
                arr = arr[0, :, -1]
            else:
                arr = arr[0]
            order = np.argsort(-np.abs(arr))[:top_k]
            return [{"feature": feature_cols[i], "impact": float(arr[i]),
                     "direction": "↑ fraud" if arr[i] > 0 else "↓ fraud"} for i in order]
        except Exception:
            pass
    # Fallback: zero-out each feature and see how the probability moves.
    base = float(get_proba(model, x)[0])
    deltas = []
    for j in range(x.shape[1]):
        xp = x.copy(); xp[0, j] = 0.0
        deltas.append(base - float(get_proba(model, xp)[0]))
    deltas = np.asarray(deltas)
    order = np.argsort(-np.abs(deltas))[:top_k]
    return [{"feature": feature_cols[i], "impact": float(deltas[i]),
             "direction": "↑ fraud" if deltas[i] > 0 else "↓ fraud"} for i in order]


def plain_english_prediction(proba, threshold, factors) -> str:
    verdict = "FRAUD" if proba >= threshold else "LEGITIMATE"
    lead = (f"This transaction is predicted **{verdict}** with a fraud probability "
            f"of **{proba:.1%}** (decision threshold {threshold:.0%}).")
    if factors:
        ups = [f["feature"] for f in factors if f["impact"] > 0][:3]
        if ups:
            lead += " The biggest fraud signals were " + ", ".join(f"`{u}`" for u in ups) + "."
    return lead


# ── Plain-English metric glossary ────────────────────────────────────────────────
METRIC_HELP = {
    "Accuracy": ("Share of all predictions that were correct.",
                 "Misleading on imbalanced data — a model can be 99% accurate by "
                 "calling everything legit. Prefer Recall / PR-AUC for fraud."),
    "Precision": ("Of the transactions we flagged as fraud, how many really were.",
                  "High precision = few false alarms to investigate."),
    "Recall": ("Of all actual frauds, how many we caught.",
               "High recall = little fraud slips through. Usually the priority."),
    "F1": ("Balance of Precision and Recall (harmonic mean).",
           "Good single number when you care about both equally."),
    "ROC-AUC": ("How well the model ranks a random fraud above a random legit txn.",
                "0.5 = coin flip, 1.0 = perfect. >0.9 is strong."),
    "PR-AUC": ("Area under the Precision-Recall curve.",
               "More informative than ROC-AUC when fraud is rare."),
    "LogLoss": ("Penalty for confident wrong probabilities (lower is better).",
                "Rewards well-calibrated probabilities."),
}
