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
                             confusion_matrix, precision_recall_curve)

from .constants import RANDOM_STATE

AUTOML_NAMES = ("FLAML", "AutoGluon")


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
def best_threshold(y_true, y_proba, beta: float = 1.0,
                   target_recall: float | None = None) -> float:
    """Pick the probability cut on the given (validation) data.

    Two strategies (the threshold is always chosen on validation only; the test
    set stays untouched):

    * **Recall-first** (``target_recall`` set, e.g. 1.0): "don't miss fraud".
      Pick the *highest* threshold that still catches at least ``target_recall``
      of the frauds — i.e. catch (almost) all fraud while blocking as few honest
      customers as possible. With target 1.0 this is the lowest fraud score, so
      every fraud in validation is caught (FN → 0), accepting more false alarms.
    * **F-beta** (``target_recall`` None): balances recall vs precision
      (beta=1 → F1). No business assumptions, well-defined for any dataset.
    """
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():          # single class → nothing to tune
        return 0.5

    if target_recall is not None:
        pos = np.asarray(y_proba)[y_true == 1]
        if len(pos) == 0:
            return 0.5
        # (1 - target_recall) quantile of fraud scores → that share of frauds may
        # fall below it; the rest (>= target_recall) are caught.
        thr = float(np.quantile(pos, max(0.0, 1.0 - float(target_recall))))
        return float(np.clip(thr, 0.005, 0.99))

    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec, rec = prec[:-1], rec[:-1]           # align with the thr array
    denom = (beta ** 2) * prec + rec
    fbeta = np.where(denom > 0, (1 + beta ** 2) * prec * rec / denom, 0.0)
    if len(thr) == 0:
        return 0.5
    return float(np.clip(thr[int(np.argmax(fbeta))], 0.01, 0.99))


def _safe(fn, *a, default=float("nan")):
    try:
        return round(fn(*a), 4)
    except Exception:
        return default


def _fit_diagnosis(train_auc, test_auc, test_pr_auc) -> tuple[str, str]:
    """Label a model as underfit / overfit / good from the train-vs-test gap.

    Logic (standard bias-variance reasoning):
      * Underfit  — even the training fit is weak (train AUC < 0.75): the model
        is too simple / features too weak; both train and test are low.
      * Overfit   — large train→test generalisation gap (> 0.10 AUC): the model
        memorised the training data and degrades on unseen data.
      * Good      — strong test performance and a small gap.
    """
    gap = (train_auc or 0) - (test_auc or 0)
    if (train_auc or 0) < 0.75 and (test_auc or 0) < 0.75:
        return "underfit", f"train AUC {train_auc:.2f} is low — model too simple / weak signal"
    if gap > 0.10:
        return "overfit", f"train−test AUC gap {gap:.2f} is large — memorising training data"
    if gap > 0.05:
        return "slight overfit", f"train−test AUC gap {gap:.2f} — mild variance, acceptable"
    return "good", f"train−test AUC gap {gap:.2f} — generalises well"


def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, beta: float = 1.0,
             target_recall: float | None = None) -> dict:
    """Tune the threshold on validation (recall-first or F-beta), report every
    metric on the untouched test set, and diagnose under/over-fitting from the
    train-vs-test ROC-AUC gap."""
    y_train = np.asarray(y_train); y_val = np.asarray(y_val); y_test = np.asarray(y_test)

    threshold = best_threshold(y_val, get_proba(model, X_val), beta, target_recall)

    proba = get_proba(model, X_test)
    pred = (proba >= threshold).astype(int)

    # Train-vs-test generalisation check (overfitting/underfitting signal).
    train_auc = _safe(roc_auc_score, y_train, get_proba(model, X_train))
    test_auc = _safe(roc_auc_score, y_test, proba)
    test_pr = _safe(average_precision_score, y_test, proba)
    status, reason = _fit_diagnosis(train_auc, test_auc, test_pr)

    return {
        "Accuracy":  _safe(accuracy_score, y_test, pred),
        "Precision": _safe(precision_score, y_test, pred, default=0.0),
        "Recall":    _safe(recall_score, y_test, pred, default=0.0),
        "F1":        _safe(f1_score, y_test, pred, default=0.0),
        "ROC-AUC":   test_auc,
        "PR-AUC":    test_pr,
        "LogLoss":   _safe(log_loss, y_test, np.clip(proba, 1e-6, 1 - 1e-6)),
        "Train AUC": train_auc,
        "Fit":       status,
        "Fit reason": reason,
        "Threshold": round(threshold, 4),
        "y_proba": proba, "y_pred": pred,
        "confusion_matrix": confusion_matrix(y_test, pred),
    }


# ── Registry ──────────────────────────────────────────────────────────────────────
# Hyper-parameters are deliberately *regularised* to keep variance in check:
# shallow trees, leaf-size floors, row/column subsampling, and L2 penalties.
# Boosters get many small trees + a learning rate and rely on early stopping
# (see ``_fit``) to choose how many to actually keep — so they neither underfit
# (too few trees) nor overfit (too many).
def build_registry(use_class_weights: bool = False, rs: int = RANDOM_STATE) -> dict:
    """Supervised models. ``use_class_weights`` MUST be False when SMOTE is used
    (otherwise imbalance is corrected twice)."""
    cw = "balanced" if use_class_weights else None
    spw = 10 if use_class_weights else 1.0

    reg = {
        # C=1.0 L2 regularisation; lbfgs is stable for this scale.
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, class_weight=cw, random_state=rs),
        # Depth + leaf floor stop the tree from memorising rare rows.
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                class_weight=cw, random_state=rs),
        # Bagging + leaf floor + feature subsampling reduce variance.
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                min_samples_leaf=5, max_features="sqrt",
                                                class_weight=cw, n_jobs=-1, random_state=rs),
        # Shallow boosted stumps + slow learning + row subsampling.
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                        max_depth=3, subsample=0.8, random_state=rs),
    }
    try:
        from xgboost import XGBClassifier
        reg["XGBoost"] = XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, gamma=0.0,
            scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, verbosity=0, random_state=rs)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        reg["LightGBM"] = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, max_depth=-1,
            min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, class_weight=cw, n_jobs=-1, verbose=-1, random_state=rs)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        reg["CatBoost"] = CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=4, l2_leaf_reg=3.0,
            verbose=0, random_seed=rs,
            auto_class_weights=("Balanced" if use_class_weights else None))
    except ImportError:
        pass

    reg["Isolation Forest"] = IsolationForestWrapper(contamination=0.05, n_jobs=-1, random_state=rs)

    base = [("lr", LogisticRegression(C=1.0, max_iter=500, class_weight=cw, random_state=rs)),
            ("dt", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight=cw, random_state=rs))]
    if "Random Forest" in reg:
        base.append(("rf", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                                   max_features="sqrt", class_weight=cw,
                                                   n_jobs=-1, random_state=rs)))
    reg["Stacking Ensemble"] = StackingClassifier(
        estimators=base, final_estimator=LogisticRegression(class_weight=cw, random_state=rs),
        cv=5, n_jobs=-1)
    reg["Voting Ensemble"] = VotingClassifier(estimators=base, voting="soft", n_jobs=-1)
    return reg


# Models that support early stopping on a validation set.
_EARLY_STOP = {"XGBoost", "LightGBM", "CatBoost"}


def _fit(model, name, Xtr, ytr, Xvl, yvl):
    """Fit a model, using early stopping on the validation set for the boosters
    so they keep just enough trees — defensive against library-version
    differences (falls back to a plain fit if the early-stopping API differs)."""
    if name == "Isolation Forest":
        return model.fit(Xtr)                         # unsupervised; ignores y
    if name in _EARLY_STOP:
        try:
            if name == "XGBoost":
                try:                                  # xgboost >= 1.6 ctor arg
                    model.set_params(early_stopping_rounds=40)
                    return model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
                except TypeError:
                    return model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
                                     early_stopping_rounds=40, verbose=False)
            if name == "LightGBM":
                import lightgbm as lgb
                return model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric="auc",
                                 callbacks=[lgb.early_stopping(40, verbose=False)])
            if name == "CatBoost":
                return model.fit(Xtr, ytr, eval_set=(Xvl, yvl),
                                 early_stopping_rounds=40, verbose=False)
        except Exception:
            pass                                      # fall through to plain fit
    return model.fit(Xtr, ytr)


def list_supervised_models() -> list[str]:
    return list(build_registry().keys())


# ── Training ──────────────────────────────────────────────────────────────────────
def train_models(selected, prep, beta: float = 1.0, progress_cb=None,
                 target_recall: float | None = None) -> dict:
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
            _fit(model, name, Xtr, ytr, Xvl, yvl)   # early stopping for boosters
            m = evaluate(model, Xtr, ytr, Xvl, yvl, Xte, yte, beta, target_recall)
            m["model"] = model
            m["train_time"] = round(time.time() - t0, 2)
            results[name] = m
        except Exception as e:
            results[name] = {"error": str(e)}
        if progress_cb:
            progress_cb((i + 1) / len(selected), name)
    return results


def train_automl(name, prep, time_limit=120, beta: float = 1.0,
                 target_recall: float | None = None):
    """Best-effort AutoML (FLAML, AutoGluon). Returns a result dict on success, or
    ``{"error": msg}`` explaining exactly why it could not run (e.g. not installed)
    so the UI can show the real reason instead of a vague 'skipped'. Scored with the
    same val/test discipline as every other model."""
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
        elif name == "AutoGluon":
            from autogluon.tabular import TabularPredictor
            tdf = pd.DataFrame(Xtr); tdf["label"] = np.asarray(ytr)
            model = TabularPredictor(label="label", eval_metric="roc_auc",
                                     verbosity=0).fit(tdf, time_limit=time_limit)
        else:
            return {"error": f"'{name}' is not a recognised AutoML option."}
        m = evaluate(model, Xtr, ytr, Xvl, yvl, Xte, yte, beta, target_recall)
        m["model"] = model
        m["train_time"] = round(time.time() - t0, 2)
        m["is_automl"] = True
        return m
    except ImportError as e:
        return {"error": f"library not installed in this Python — {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ── Best-model selection ─────────────────────────────────────────────────────────
def pick_best_model(results: dict) -> str | None:
    """Choose the best model — recall-first, the way a fraud team would.

    The priority is **catch the most fraud** (highest Recall), then **bother the
    fewest honest customers** (highest Precision), then ranking quality (PR-AUC)
    and calibration (lowest LogLoss). Metrics are rounded to 4 decimals before
    comparison so two genuinely-tied models don't flip the winner on float noise —
    so the chosen model is **stable** across identical training runs.
    """
    def _ok(r):
        return ("error" not in r and all(
            isinstance(r.get(k), (int, float)) and not np.isnan(r.get(k))
            for k in ("Recall", "Precision", "PR-AUC", "LogLoss")))

    cand = {n: r for n, r in results.items() if _ok(r)}
    if not cand:                                   # fallback: any usable ROC-AUC
        roc = {n: r for n, r in results.items()
               if isinstance(r.get("ROC-AUC"), (int, float)) and not np.isnan(r["ROC-AUC"])}
        return max(roc, key=lambda n: roc[n]["ROC-AUC"], default=None)

    def key(n):
        r = cand[n]
        return (round(r["Recall"], 4), round(r["Precision"], 4),
                round(r["PR-AUC"], 4), -round(r["LogLoss"], 4), n)
    # Sort by the tuple (incl. name as a final deterministic tiebreak), take the top.
    return sorted(cand, key=key, reverse=True)[0]


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


# ── Detection summary ──────────────────────────────────────────────────────────
def detection_summary(y_true, y_pred) -> dict:
    """Plain confusion-matrix outcomes on the test set — no cost assumptions.

    Reports the four cells plus the two rates that matter for fraud:
      * detection rate (recall)  — share of real fraud we caught
      * false-alarm rate         — share of legit txns wrongly flagged
    Everything here is directly observed on the held-out test set, so it needs
    no fabricated dollar figures to be meaningful.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    pos = tp + fn
    neg = tn + fp
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "detection_rate": (tp / pos) if pos else 0.0,        # recall
        "false_alarm_rate": (fp / neg) if neg else 0.0,
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
    }
