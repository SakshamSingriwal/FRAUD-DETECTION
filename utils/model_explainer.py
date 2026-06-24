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
            # TreeExplainer covers CatBoost / LightGBM / XGBoost / sklearn trees and
            # ensembles, and needs no background data — so its values aren't all
            # zero (the old model-agnostic path used the single instance as its own
            # background, which collapses every contribution to 0).
            arr = None
            try:
                vals = shap.TreeExplainer(model)(x)
                a = np.asarray(vals.values)
                a = a[0, :, -1] if a.ndim == 3 else a[0]
                if np.abs(a).sum() > 1e-12:
                    arr = a
            except Exception:
                arr = None
            if arr is not None:
                order = np.argsort(-np.abs(arr))[:top_k]
                return [{"feature": feature_cols[i], "impact": float(arr[i]),
                         "direction": "↑ fraud" if arr[i] > 0 else "↓ fraud"} for i in order]
        except Exception:
            pass
    # Linear models (e.g. Logistic Regression): the exact local contribution to the
    # log-odds is coef × feature value — far better than perturbing a saturated
    # sigmoid (which barely moves and yields ~0 everywhere).
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        if coef.shape[0] == x.shape[1]:
            contrib = coef * x[0]
            if np.abs(contrib).sum() > 1e-12:
                order = np.argsort(-np.abs(contrib))[:top_k]
                return [{"feature": feature_cols[i], "impact": float(contrib[i]),
                         "direction": "↑ fraud" if contrib[i] > 0 else "↓ fraud"} for i in order]

    # Final fallback (other models): zero-out each feature and see how the
    # probability moves.
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
# Each entry is (what it means, why it matters, what a good value looks like) so a
# beginner can both read the result AND explain it to someone else.
METRIC_HELP = {
    "Accuracy": ("Share of all predictions that were correct: (TP+TN) / everything.",
                 "Misleading on imbalanced data — a model can be 99% accurate by calling "
                 "everything legit. Prefer Recall / PR-AUC for fraud.",
                 "High looks nice but ignore it when fraud is rare."),
    "Precision": ("Of the transactions we flagged as fraud, how many really were: TP / (TP+FP).",
                  "High precision = few false alarms, so investigators don't waste time.",
                  ">0.5 is workable; closer to 1.0 means cleaner alerts."),
    "Recall": ("Of all the real frauds, how many we caught: TP / (TP+FN). Also called "
               "sensitivity or the true-positive rate.",
               "High recall = little fraud slips through. Usually the top priority in fraud.",
               ">0.8 is good; the cost of a missed fraud (FN) is normally high."),
    "F1": ("The harmonic mean of Precision and Recall — a single score that is only high "
           "when BOTH are high.",
           "Best single number when you care about catching fraud and avoiding false alarms.",
           ">0.7 is solid on imbalanced data."),
    "ROC-AUC": ("Probability the model ranks a random fraud above a random legit transaction "
                "(area under the TPR-vs-FPR curve). Threshold-independent.",
                "Measures ranking quality regardless of the cutoff you pick.",
                "0.5 = coin flip · 0.8 good · >0.9 strong · 1.0 = perfect (suspicious)."),
    "PR-AUC": ("Area under the Precision-Recall curve (a.k.a. average precision).",
               "More honest than ROC-AUC when fraud is rare, because it ignores the easy "
               "true negatives and focuses on the positive (fraud) class.",
               "Compare against the fraud rate — a 3% fraud rate makes 0.5 PR-AUC strong."),
    "LogLoss": ("Penalty for confident-but-wrong probabilities (lower is better).",
                "Rewards well-calibrated probabilities, not just correct yes/no calls.",
                "Lower is better; ~0 is perfect, 0.69 ≈ random guessing."),
    "Train AUC": ("ROC-AUC measured on the data the model was trained on.",
                  "Compared against test ROC-AUC, it reveals over/under-fitting.",
                  "Should be close to test AUC — a big gap means overfitting."),
    "Fit": ("Verdict from the train→test AUC gap: underfit / good / overfit.",
            "Tells you whether the model is too simple, just right, or memorising.",
            "Aim for 'good' (small gap, strong test score)."),
    "Threshold": ("The probability cut-off that turns a score into a fraud/legit decision.",
                  "Sentinel picks the value that maximises F1 on the validation split.",
                  "Lower → catch more fraud but more false alarms; higher → the reverse."),
}
