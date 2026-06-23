"""
model_trainer.py — Supervised model registry, unified probability access,
threshold-disciplined evaluation, AutoML wrappers, and business-impact math.

Correctness invariants (from the production hardening work):
  * ``get_proba`` is the SINGLE probability entry point. It NEVER returns silent
    zeros — it raises if a model cannot score.
  * The Isolation Forest is wrapped so its anomaly score is scaled with
    TRAIN-time min/max (no per-batch collapse on single rows).
  * The threshold is tuned on VALIDATION and every metric is reported on TEST.
  * Imbalance is corrected ONCE: class weights are used only when SMOTE is off.
  * Every stochastic estimator is seeded.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier, VotingClassifier, IsolationForest)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, log_loss,
                             confusion_matrix)

from .constants import RANDOM_STATE

AUTOML_NAMES = ("H2O AutoML", "AutoGluon", "FLAML")


# ── Unified probability access ──────────────────────────────────────────────────
def get_proba(model, X) -> np.ndarray:
    """Positive-class probability for every supported model. Raises instead of
    returning zeros so a non-scoring model can never silently read as 0% fraud."""
    mod = type(model).__module__ or ""

    if mod.startswith("h2o"):
        import h2o
        preds = model.predict(h2o.H2OFrame(pd.DataFrame(np.asarray(X)))).as_data_frame()
        if "p1" in preds.columns:
            return preds["p1"].to_numpy()
        cols = [c for c in preds.columns if c != "predict"]
        if cols:
            return preds[cols[-1]].to_numpy()
        raise ValueError("H2O model returned no probability column.")

    if mod.startswith("autogluon"):
        proba = model.predict_proba(pd.DataFrame(np.asarray(X)))
        return (proba[1] if 1 in proba.columns else proba.iloc[:, -1]).to_numpy()

    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()

    raise ValueError(
        f"{type(model).__name__} cannot produce probabilities "
        "(no predict_proba / recognised AutoML interface)."
    )


class IsolationForestWrapper:
    """IsolationForest with a standard ``predict_proba``. Anomaly scores are
    min-max scaled using values stored at fit() time, fixing the single-row
    collapse bug. Picklable (top-level class)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("random_state", RANDOM_STATE)
        self.model = IsolationForest(**kwargs)
        self._lo = self._hi = None

    def fit(self, X, y=None):
        self.model.fit(X)
        s = -self.model.decision_function(X)
        self._lo, self._hi = float(np.min(s)), float(np.max(s))
        return self

    def _scaled(self, X):
        if self._lo is None:
            raise RuntimeError("IsolationForestWrapper used before fit().")
        s = -self.model.decision_function(X)
        return np.clip((s - self._lo) / ((self._hi - self._lo) or 1e-9), 0, 1)

    def predict_proba(self, X):
        p = self._scaled(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._scaled(X) >= 0.5).astype(int)


# ── Threshold & evaluation ───────────────────────────────────────────────────────
def find_optimal_threshold(y_true, y_proba, cost_fn=1.0, cost_fp=0.1):
    y_true = np.asarray(y_true)
    best_t, best_c = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 200):
        pred = (y_proba >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        c = cost_fn * fn + cost_fp * fp
        if c < best_c:
            best_c, best_t = c, t
    return best_t, best_c


def _safe(fn, *a, default=float("nan")):
    try:
        return round(fn(*a), 4)
    except Exception:
        return default


def evaluate(model, X_val, y_val, X_test, y_test, cost_fn=1.0, cost_fp=0.1) -> dict:
    """Tune threshold on validation, report all metrics on the untouched test."""
    y_val, y_test = np.asarray(y_val), np.asarray(y_test)
    val_proba = get_proba(model, X_val)
    threshold, _ = find_optimal_threshold(y_val, val_proba, cost_fn, cost_fp)

    proba = get_proba(model, X_test)
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy":  _safe(accuracy_score, y_test, pred),
        "Precision": _safe(precision_score, y_test, pred, default=0.0),
        "Recall":    _safe(recall_score, y_test, pred, default=0.0),
        "F1":        _safe(f1_score, y_test, pred, default=0.0),
        "ROC-AUC":   _safe(roc_auc_score, y_test, proba),
        "PR-AUC":    _safe(average_precision_score, y_test, proba),
        "LogLoss":   _safe(log_loss, y_test, np.clip(proba, 1e-6, 1 - 1e-6)),
        "Threshold": round(threshold, 4),
        "y_proba": proba, "y_pred": pred,
        "confusion_matrix": confusion_matrix(y_test, pred),
    }


# ── Registry ──────────────────────────────────────────────────────────────────────
def build_registry(use_class_weights: bool = False, rs: int = RANDOM_STATE) -> dict:
    """Supervised models. ``use_class_weights`` MUST be False when SMOTE is used
    (otherwise imbalance is corrected twice)."""
    cw = "balanced" if use_class_weights else None
    spw = 10 if use_class_weights else 1.0

    reg = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw, random_state=rs),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight=cw, random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=-1, random_state=rs),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, random_state=rs),
    }
    try:
        from xgboost import XGBClassifier
        reg["XGBoost"] = XGBClassifier(n_estimators=200, scale_pos_weight=spw,
                                       eval_metric="logloss", n_jobs=-1, verbosity=0, random_state=rs)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        reg["LightGBM"] = LGBMClassifier(n_estimators=200, class_weight=cw, n_jobs=-1,
                                         verbose=-1, random_state=rs)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        reg["CatBoost"] = CatBoostClassifier(iterations=200, verbose=0, random_seed=rs,
                                             auto_class_weights=("Balanced" if use_class_weights else None))
    except ImportError:
        pass

    reg["Isolation Forest"] = IsolationForestWrapper(contamination=0.05, n_jobs=-1, random_state=rs)

    base = [("lr", LogisticRegression(max_iter=500, class_weight=cw, random_state=rs)),
            ("dt", DecisionTreeClassifier(max_depth=6, class_weight=cw, random_state=rs))]
    if "Random Forest" in reg:
        base.append(("rf", RandomForestClassifier(n_estimators=100, class_weight=cw,
                                                   n_jobs=-1, random_state=rs)))
    reg["Stacking Ensemble"] = StackingClassifier(
        estimators=base, final_estimator=LogisticRegression(class_weight=cw, random_state=rs), n_jobs=-1)
    reg["Voting Ensemble"] = VotingClassifier(estimators=base, voting="soft", n_jobs=-1)
    return reg


def list_supervised_models() -> list[str]:
    return list(build_registry().keys())


# ── Training ──────────────────────────────────────────────────────────────────────
def train_models(selected, prep, cost_fn=1.0, cost_fp=0.1, progress_cb=None) -> dict:
    use_cw = not prep.get("apply_smote", False)
    reg = build_registry(use_class_weights=use_cw)
    Xtr, ytr = prep["X_train"], prep["y_train"]
    Xvl, yvl = prep["X_val"], prep["y_val"]
    Xte, yte = prep["X_test"], prep["y_test"]

    results = {}
    for i, name in enumerate(selected):
        if name not in reg:
            continue
        model = reg[name]
        t0 = time.time()
        try:
            if name == "Isolation Forest":
                model.fit(Xtr)            # unsupervised; wrapper stores scaling
            else:
                model.fit(Xtr, ytr)
            m = evaluate(model, Xvl, yvl, Xte, yte, cost_fn, cost_fp)
            m["model"] = model
            m["train_time"] = round(time.time() - t0, 2)
            results[name] = m
        except Exception as e:
            results[name] = {"error": str(e)}
        if progress_cb:
            progress_cb((i + 1) / len(selected), name)
    return results


def train_automl(name, prep, time_limit=120, cost_fn=1.0, cost_fp=0.1):
    """Best-effort AutoML. Returns a result dict or None if the framework is not
    installed / fails. Scored with the same val/test discipline."""
    Xtr, ytr = prep["X_train"], prep["y_train"]
    Xvl, yvl = prep["X_val"], prep["y_val"]
    Xte, yte = prep["X_test"], prep["y_test"]
    t0 = time.time()
    try:
        if name == "FLAML":
            from flaml import AutoML
            model = AutoML()
            model.fit(np.asarray(Xtr), np.asarray(ytr), task="classification",
                      time_budget=time_limit, metric="roc_auc", verbose=0, seed=RANDOM_STATE)
        elif name == "H2O AutoML":
            import h2o
            from h2o.automl import H2OAutoML
            h2o.init(verbose=False)
            tdf = pd.DataFrame(Xtr); tdf["label"] = np.asarray(ytr)
            hf = h2o.H2OFrame(tdf); hf["label"] = hf["label"].asfactor()
            aml = H2OAutoML(max_runtime_secs=time_limit, seed=RANDOM_STATE, verbosity=None)
            aml.train(y="label", training_frame=hf)
            model = aml.leader
        elif name == "AutoGluon":
            from autogluon.tabular import TabularPredictor
            tdf = pd.DataFrame(Xtr); tdf["label"] = np.asarray(ytr)
            model = TabularPredictor(label="label", eval_metric="roc_auc",
                                     verbosity=0).fit(tdf, time_limit=time_limit)
        else:
            return None
        m = evaluate(model, Xvl, yvl, Xte, yte, cost_fn, cost_fp)
        m["model"] = model
        m["train_time"] = round(time.time() - t0, 2)
        m["is_automl"] = True
        return m
    except Exception:
        return None


# ── Persistence ──────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class ModelNotPersistableError(RuntimeError):
    """Raised when an AutoML model cannot be joblib-pickled."""


def save_artifacts(model, scaler, feature_cols, model_name, extra=None):
    """Persist a non-AutoML model + its scaler/features. AutoML leaders live in
    their own runtime and are not joblib-safe, so we refuse and let the caller
    keep them for the session only."""
    if model_name in AUTOML_NAMES or (type(model).__module__ or "").split(".")[0] in ("h2o", "autogluon", "flaml"):
        raise ModelNotPersistableError(
            f"'{model_name}' is an AutoML model and is not saved to disk "
            "(available for this session only).")
    safe = model_name.replace(" ", "_")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{safe}.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump({"name": model_name, **(extra or {})}, os.path.join(MODELS_DIR, "meta.pkl"))


def load_artifacts():
    meta_path = os.path.join(MODELS_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    meta = joblib.load(meta_path)
    safe = meta["name"].replace(" ", "_")
    return {
        "model": joblib.load(os.path.join(MODELS_DIR, f"{safe}.pkl")),
        "scaler": joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        "feature_cols": joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl")),
        "meta": meta,
    }


# ── Business impact ──────────────────────────────────────────────────────────────
def business_impact(y_true, y_pred, amounts=None, fp_review_cost=5.0,
                    avg_fraud_amount=None) -> dict:
    """Translate a confusion matrix into dollars.

    ``amounts`` (per-row $) is used when available; otherwise a flat
    ``avg_fraud_amount`` is assumed. Net savings = fraud caught − FP review cost.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = (y_pred == 1) & (y_true == 1)
    fn = (y_pred == 0) & (y_true == 1)
    fp = (y_pred == 1) & (y_true == 0)

    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        caught = float(amounts[tp].sum())
        missed = float(amounts[fn].sum())
    else:
        avg = avg_fraud_amount or 1000.0
        caught = float(tp.sum() * avg)
        missed = float(fn.sum() * avg)

    fp_cost = float(fp.sum() * fp_review_cost)
    net = caught - fp_cost
    return {
        "fraud_caught": caught, "fraud_missed": missed,
        "fp_count": int(fp.sum()), "fp_cost": fp_cost,
        "net_savings": net,
        "roi_pct": round((net / fp_cost * 100), 1) if fp_cost > 0 else float("inf"),
        "n_tp": int(tp.sum()), "n_fn": int(fn.sum()),
    }
