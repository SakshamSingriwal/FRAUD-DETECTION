"""
utils.py — Helper functions for Fraud Detection Studio.
Supports: automatic feature engineering OR manual column selection,
user-selectable target column, SMOTE, threshold tuning, AutoML wrappers.

This module has been hardened for production correctness. See CHANGELOG.md
for the full list of fixes. Key invariants enforced here:

* Probabilities are produced through a single ``get_proba()`` entry point
  that NEVER silently returns zeros — it raises if a model cannot score.
* Isolation Forest anomaly scores are scaled with parameters learned at
  TRAINING time (via ``IsolationForestWrapper``), not per-batch.
* The decision threshold is tuned on a dedicated VALIDATION split and the
  reported metrics come from an untouched TEST split.
* Class imbalance is corrected ONCE (SMOTE *or* class weights, never both).
* Every stochastic model is seeded for reproducibility.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# Global seed used everywhere so results are reproducible across runs.
RANDOM_STATE = 42

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

REQUIRED_COLS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
]
TRAIN_COLS = REQUIRED_COLS + ["isFraud"]
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Model names that are AutoML predictors. These objects are NOT joblib-safe
# (H2O lives in a JVM cluster; AutoGluon persists to its own directory), so we
# keep them out of the disk save/load path and warn the caller instead.
AUTOML_MODEL_NAMES = ("H2O AutoML", "AutoGluon")


# ── Feature Engineering (Automatic mode) ──────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build PaySim domain features. Defensive against missing values so the
    same code path works for training *and* single/batch prediction."""
    df = df.copy()

    # Coerce the numeric balance/amount columns and fill gaps so downstream
    # arithmetic never produces NaN (which would silently poison the scaler).
    numeric_cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # Categorical/string inputs used below.
    if "nameDest" not in df.columns:
        df["nameDest"] = ""
    df["nameDest"] = df["nameDest"].astype(str)
    if "type" not in df.columns:
        df["type"] = ""

    df["error_balance_orig"]    = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["error_balance_dest"]    = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["is_origin_emptied"]     = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
    df["dest_initial_zero"]     = (df["oldbalanceDest"] == 0).astype(int)
    df["is_merchant"]           = df["nameDest"].str.startswith("M").astype(int)
    df["is_cash_out_transfer"]  = df["type"].isin(["CASH_OUT", "TRANSFER"]).astype(int)
    df["amount_log"]            = np.log1p(df["amount"])
    df["orig_balance_change"]   = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_change"]   = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_balance_ratio"]    = df["newbalanceOrig"] / (df["oldbalanceOrg"] + 1)
    df["dest_balance_ratio"]    = df["newbalanceDest"] / (df["oldbalanceDest"] + 1)
    df["amount_to_orig_ratio"]  = df["amount"] / (df["oldbalanceOrg"] + 1)
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)
    df.drop(columns=[c for c in ["nameOrig", "nameDest", "step", "type"] if c in df.columns], inplace=True)
    return df


def find_correlated_features(df, threshold=0.85, target_col="isFraud"):
    """Return the list of columns to drop because they are highly correlated
    with another feature. Correlation is computed on whatever ``df`` is passed
    — callers pass the TRAINING split only, to avoid leakage."""
    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) < 2:
        return []
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [col for col in upper.columns if any(upper[col] > threshold)]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


# ── Probability access (single entry point for every model) ───────────────────

def _is_automl_model(model) -> bool:
    mod = type(model).__module__ or ""
    return mod.startswith("h2o") or mod.startswith("autogluon")


def get_proba(model, X) -> np.ndarray:
    """Unified positive-class probability extractor for every supported model.

    Order of resolution:
      1. H2O leader            → predict() → ``p1`` column
      2. AutoGluon predictor   → predict_proba() positive column
      3. anything with predict_proba (sklearn, xgb, lgbm, catboost, the
         Isolation Forest wrapper, stacking)
      4. otherwise             → raise (NEVER return silent zeros)
    """
    mod = type(model).__module__ or ""

    # 1. H2O AutoML leader.
    if mod.startswith("h2o"):
        import h2o
        hf = h2o.H2OFrame(pd.DataFrame(np.asarray(X)))
        preds = model.predict(hf).as_data_frame()
        if "p1" in preds.columns:
            return preds["p1"].to_numpy()
        prob_cols = [c for c in preds.columns if c != "predict"]
        if prob_cols:
            return preds[prob_cols[-1]].to_numpy()
        raise ValueError("H2O model returned no probability column.")

    # 2. AutoGluon TabularPredictor (predict_proba returns a DataFrame).
    if mod.startswith("autogluon"):
        proba_df = model.predict_proba(pd.DataFrame(np.asarray(X)))
        if 1 in proba_df.columns:
            return proba_df[1].to_numpy()
        return proba_df.iloc[:, -1].to_numpy()

    # 3. Standard probabilistic models (incl. IsolationForestWrapper).
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    # 4. No usable scoring method — fail loudly rather than emit zeros.
    raise ValueError(
        f"Model of type {type(model).__name__} cannot produce probabilities "
        "(no predict_proba / recognised AutoML interface)."
    )


# Backwards-compatible alias (app.py and older code import ``_get_proba``).
_get_proba = get_proba


# ── Isolation Forest wrapper (train-time scaling, picklable) ───────────────────

class IsolationForestWrapper:
    """Wraps :class:`sklearn.ensemble.IsolationForest` so it exposes a standard
    ``predict_proba``.

    The raw anomaly score is min-max scaled using the min/max observed on the
    TRAINING set (stored at ``fit`` time). This fixes the bug where per-batch
    scaling collapsed a single-row prediction to 0 (min == max)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("random_state", RANDOM_STATE)
        self.model = IsolationForest(**kwargs)
        self._score_min = None
        self._score_max = None

    def fit(self, X, y=None):
        self.model.fit(X)
        scores = -self.model.decision_function(X)      # higher == more anomalous
        self._score_min = float(np.min(scores))
        self._score_max = float(np.max(scores))
        return self

    def _scaled_scores(self, X):
        if self._score_min is None:
            raise RuntimeError("IsolationForestWrapper used before fit().")
        scores = -self.model.decision_function(X)
        denom = (self._score_max - self._score_min) or 1e-9
        return np.clip((scores - self._score_min) / denom, 0.0, 1.0)

    def predict_proba(self, X):
        p1 = self._scaled_scores(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self._scaled_scores(X) >= 0.5).astype(int)


# ── Preprocessing Pipeline ────────────────────────────────────────────────────

def preprocess(
    train_df,
    test_size=0.2,
    val_size=0.2,
    corr_threshold=0.85,
    apply_smote=True,
    random_state=RANDOM_STATE,
    target_col="isFraud",
    feature_mode="automatic",
    manual_feature_cols=None,
    apply_corr_removal=True,
):
    """Build train / validation / test splits with no leakage.

    Pipeline order (each step learns ONLY from training data):
      1. feature engineering / manual selection
      2. split off the TEST set
      3. split off the VALIDATION set from the remaining train data
      4. correlation-based feature removal — decided on TRAIN only
      5. StandardScaler — fitted on TRAIN only
      6. SMOTE — applied to TRAIN only (never val/test)
    """
    df = train_df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    if feature_mode == "automatic":
        df_feat = engineer_features(df)
        if target_col not in df_feat.columns:
            df_feat[target_col] = df[target_col].values
    else:
        cols     = [c for c in (list(manual_feature_cols or []) + [target_col]) if c in df.columns]
        df_feat  = df[cols].copy()
        feat_raw = encode_categoricals(df_feat.drop(columns=[target_col]))
        df_feat  = feat_raw.copy()
        df_feat[target_col] = df[target_col].values

    feature_cols_all = [c for c in df_feat.columns if c != target_col]
    X = df_feat[feature_cols_all].astype(float)
    y = df_feat[target_col]

    if y.nunique() < 2:
        raise ValueError(
            "Target column has a single class — cannot train a fraud classifier. "
            "Check that the dataset contains both legitimate and fraudulent rows."
        )

    # 2. Hold out the test set first.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Carve a validation set out of the remaining training data. The
    #    validation set is used ONLY to tune the decision threshold.
    stratify_inner = y_train_full if y_train_full.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size,
        random_state=random_state, stratify=stratify_inner,
    )

    # 4. Correlation removal decided on the training split only (no leakage).
    if apply_corr_removal:
        dropped_corr = find_correlated_features(
            pd.concat([X_train, y_train], axis=1), corr_threshold, target_col
        )
    else:
        dropped_corr = []
    feature_cols = [c for c in feature_cols_all if c not in dropped_corr]
    X_train, X_val, X_test = X_train[feature_cols], X_val[feature_cols], X_test[feature_cols]

    # 5. Scale (fit on train only).
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # 6. SMOTE on the training set only.
    smote_applied = bool(apply_smote and y_train.sum() > 0)
    if smote_applied:
        sm = SMOTE(random_state=random_state)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    return {
        "X_train":       X_train_s,
        "X_val":         X_val_s,
        "X_test":        X_test_s,
        "y_train":       y_train,
        "y_val":         y_val,
        "y_test":        y_test,
        "scaler":        scaler,
        "feature_cols":  feature_cols,
        "dropped_corr":  dropped_corr,
        "df_engineered": df_feat,
        "target_col":    target_col,
        "feature_mode":  feature_mode,
        "apply_smote":   smote_applied,
    }


# ── Threshold & Metrics ───────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_proba, cost_fn=1.0, cost_fp=0.1):
    """Pick the threshold minimising cost = cost_fn*FN + cost_fp*FP."""
    y_true = np.asarray(y_true)
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh, best_cost = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_thresh = cost, t
    return best_thresh, best_cost


def _safe_auc(y_true, y_proba):
    try:
        return round(roc_auc_score(y_true, y_proba), 4)
    except ValueError:
        return float("nan")


def evaluate_model(model, X_val, y_val, X_test, y_test, cost_fn=1.0, cost_fp=0.1):
    """Tune the threshold on the VALIDATION set, then report every metric on the
    untouched TEST set. This removes the leakage of tuning and scoring on the
    same data."""
    y_val  = np.asarray(y_val)
    y_test = np.asarray(y_test)

    # 1. Tune threshold on validation predictions only.
    val_proba = get_proba(model, X_val)
    threshold, _ = find_optimal_threshold(y_val, val_proba, cost_fn, cost_fp)

    # 2. Evaluate on the held-out test set.
    y_proba = get_proba(model, X_test)
    y_pred  = (y_proba >= threshold).astype(int)
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    test_cost = cost_fn * fn + cost_fp * fp

    return {
        "Accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "Precision":         round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":            round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":                round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":           _safe_auc(y_test, y_proba),
        "Optimal Threshold": round(threshold, 4),
        "Min Cost":          round(test_cost, 2),
        "y_proba":           y_proba,
        "y_pred":            y_pred,
        "confusion_matrix":  confusion_matrix(y_test, y_pred),
    }


# ── Model Registry & Training ─────────────────────────────────────────────────

def build_model_registry(use_class_weights=False, random_state=RANDOM_STATE, time_limit=300):
    """Create the model registry.

    ``use_class_weights`` controls whether class-imbalance weighting is applied.
    It MUST be False when SMOTE is used, otherwise imbalance is corrected twice
    (over-predicting fraud). The training flow sets this automatically.

    Every stochastic estimator is seeded with ``random_state`` for
    reproducibility.
    """
    cw  = "balanced" if use_class_weights else None
    spw = 10 if use_class_weights else 1.0       # XGBoost scale_pos_weight

    registry = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=cw, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight=cw, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight=cw, n_jobs=-1, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier
        registry["XGBoost"] = XGBClassifier(
            n_estimators=100, scale_pos_weight=spw,
            eval_metric="logloss", n_jobs=-1, verbosity=0,
            random_state=random_state)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        registry["LightGBM"] = LGBMClassifier(
            n_estimators=100, class_weight=cw, n_jobs=-1, verbose=-1,
            random_state=random_state)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        registry["CatBoost"] = CatBoostClassifier(
            iterations=100, verbose=0, random_seed=random_state,
            auto_class_weights=("Balanced" if use_class_weights else None))
    except ImportError:
        pass

    registry["Isolation Forest"] = IsolationForestWrapper(
        contamination=0.01, n_jobs=-1, random_state=random_state)

    estimators = [
        ("lr", LogisticRegression(max_iter=500, class_weight=cw, random_state=random_state)),
        ("dt", DecisionTreeClassifier(max_depth=6, class_weight=cw, random_state=random_state)),
    ]
    try:
        from xgboost import XGBClassifier as _XGB
        estimators.append(("xgb", _XGB(
            n_estimators=50, scale_pos_weight=spw,
            eval_metric="logloss", verbosity=0, random_state=random_state)))
    except ImportError:
        pass
    registry["Stacking Ensemble"] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight=cw, random_state=random_state),
        n_jobs=-1,
    )
    return registry


def train_models(selected, X_train, y_train, X_val, y_val, X_test, y_test,
                 use_class_weights=False, time_limit=300, progress_cb=None):
    registry = build_model_registry(use_class_weights=use_class_weights, time_limit=time_limit)
    results  = {}
    for i, name in enumerate(selected):
        if name not in registry:
            continue
        model = registry[name]
        try:
            if name == "Isolation Forest":
                model.fit(X_train)          # unsupervised; wrapper stores scaling
            else:
                model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
            metrics["model"] = model
            results[name] = metrics
        except Exception as e:
            results[name] = {"error": str(e)}
        if progress_cb:
            progress_cb((i + 1) / len(selected), f"✅ Trained: {name}")
    return results


# ── AutoML Wrappers ───────────────────────────────────────────────────────────

def train_h2o_automl(X_train, y_train, X_val, y_val, X_test, y_test, time_limit=300):
    try:
        import h2o
        from h2o.automl import H2OAutoML
        h2o.init(verbose=False)
        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train.values if hasattr(y_train, "values") else y_train
        h_train = h2o.H2OFrame(train_df)
        h_train["label"] = h_train["label"].asfactor()
        aml = H2OAutoML(max_runtime_secs=time_limit, seed=RANDOM_STATE, verbosity=None)
        aml.train(y="label", training_frame=h_train)
        # Tune threshold on validation, score on test — via the shared get_proba.
        return _automl_metrics(aml.leader, X_val, y_val, X_test, y_test, is_h2o=True)
    except Exception:
        return None


def train_autogluon(X_train, y_train, X_val, y_val, X_test, y_test, time_limit=300):
    try:
        from autogluon.tabular import TabularPredictor
        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train.values if hasattr(y_train, "values") else y_train
        predictor = TabularPredictor(label="label", eval_metric="roc_auc", verbosity=0).fit(
            train_df, time_limit=time_limit)
        return _automl_metrics(predictor, X_val, y_val, X_test, y_test, is_autogluon=True)
    except Exception:
        return None


def _automl_metrics(model, X_val, y_val, X_test, y_test, is_h2o=False, is_autogluon=False):
    """Shared metric computation for AutoML leaders: threshold tuned on
    validation, metrics reported on test (same discipline as evaluate_model)."""
    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
    metrics["model"]         = model
    metrics["is_h2o"]        = is_h2o
    metrics["is_autogluon"]  = is_autogluon
    return metrics


# ── Save / Load Artefacts ─────────────────────────────────────────────────────

class ModelNotPersistableError(RuntimeError):
    """Raised when an AutoML model cannot be saved with joblib."""


def save_artefacts(model, scaler, feature_cols, model_name,
                   target_col="isFraud", feature_mode="automatic"):
    """Persist the trained artefacts. AutoML models (H2O / AutoGluon) are not
    joblib-safe, so we refuse to save them and let the caller decide how to
    surface that (the app keeps them for the current session only)."""
    if model_name in AUTOML_MODEL_NAMES or _is_automl_model(model):
        raise ModelNotPersistableError(
            f"'{model_name}' is an AutoML model and cannot be persisted to disk. "
            "It remains available for the current session only."
        )

    safe = model_name.replace(" ", "_")
    joblib.dump(model,        os.path.join(MODELS_DIR, f"{safe}.pkl"))
    joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(model_name,   os.path.join(MODELS_DIR, "best_model_name.pkl"))
    joblib.dump({"target_col": target_col, "feature_mode": feature_mode},
                os.path.join(MODELS_DIR, "config.pkl"))


def load_artefacts():
    name_path = os.path.join(MODELS_DIR, "best_model_name.pkl")
    if not os.path.exists(name_path):
        return None, None, None, None, "isFraud", "automatic"
    model_name   = joblib.load(name_path)
    safe         = model_name.replace(" ", "_")
    model        = joblib.load(os.path.join(MODELS_DIR, f"{safe}.pkl"))
    scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    cfg_path     = os.path.join(MODELS_DIR, "config.pkl")
    cfg = joblib.load(cfg_path) if os.path.exists(cfg_path) else {}
    return (model, scaler, feature_cols, model_name,
            cfg.get("target_col", "isFraud"), cfg.get("feature_mode", "automatic"))


# ── Prediction Helpers ────────────────────────────────────────────────────────

def prepare_input_for_prediction(raw_df, scaler, feature_cols,
                                  feature_mode="automatic", manual_raw_cols=None):
    """Transform raw input into the exact feature matrix the model expects.

    Guarantees the same columns, order, and dtypes as training. Any column the
    model expects but the input lacks is filled with 0; any NaN is filled with 0
    so the scaler never receives missing values (which would crash or skew)."""
    if feature_mode == "automatic":
        df = engineer_features(raw_df)
    else:
        cols = [c for c in (manual_raw_cols or []) if c in raw_df.columns]
        df   = encode_categoricals(raw_df[cols].copy())

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return scaler.transform(X)


def predict_batch(raw_df, model, scaler, feature_cols, threshold=0.5,
                  feature_mode="automatic", manual_raw_cols=None):
    X     = prepare_input_for_prediction(raw_df, scaler, feature_cols, feature_mode, manual_raw_cols)
    proba = get_proba(model, X)
    out   = raw_df.copy()
    out["fraud_probability"] = proba
    out["fraud_prediction"]  = (proba >= threshold).astype(int)
    return out


# ── Plot Helpers ──────────────────────────────────────────────────────────────

_DARK = "#0f1117"
_CARD = "#1e2130"
_BORDER = "#2d3250"
_TEXT  = "#9ca3af"
_PALETTE = ["#4A90D9","#E53935","#43A047","#FB8C00","#AB47BC",
            "#00ACC1","#F06292","#8D6E63","#78909C","#26A69A"]


def _dark_ax(ax, fig=None):
    if fig:
        fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_DARK)
    ax.tick_params(colors=_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=_DARK)
    _dark_ax(ax)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"],
                ax=ax, linewidths=0.5, linecolor=_CARD)
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.set_ylabel("Actual", color=_TEXT)
    ax.set_xlabel("Predicted", color=_TEXT)
    ax.tick_params(colors=_TEXT)
    plt.tight_layout()
    return fig


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=_DARK)
    _dark_ax(ax, fig)
    for i, (name, res) in enumerate(results.items()):
        if "y_proba" not in res:
            continue
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, color=_PALETTE[i % len(_PALETTE)],
                label=f"{name} (AUC={res.get('ROC-AUC',0):.3f})", linewidth=2)
    ax.plot([0,1],[0,1],"w--",alpha=0.3)
    ax.set_xlabel("False Positive Rate", color=_TEXT)
    ax.set_ylabel("True Positive Rate", color=_TEXT)
    ax.set_title("ROC Curves", color="white", fontsize=12)
    ax.legend(loc="lower right", fontsize=7, facecolor=_CARD, labelcolor="white")
    plt.tight_layout()
    return fig


def plot_pr_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=_DARK)
    _dark_ax(ax, fig)
    for i, (name, res) in enumerate(results.items()):
        if "y_proba" not in res:
            continue
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ap = average_precision_score(y_test, res["y_proba"])
        ax.plot(rec, prec, color=_PALETTE[i % len(_PALETTE)],
                label=f"{name} (AP={ap:.3f})", linewidth=2)
    ax.set_xlabel("Recall", color=_TEXT)
    ax.set_ylabel("Precision", color=_TEXT)
    ax.set_title("Precision-Recall Curves", color="white", fontsize=12)
    ax.legend(loc="upper right", fontsize=7, facecolor=_CARD, labelcolor="white")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax    = plt.subplots(figsize=(10, 8), facecolor=_DARK)
    _dark_ax(ax, fig)
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, ax=ax, linewidths=0.3, linecolor=_CARD)
    ax.set_title("Correlation Heatmap", color="white", fontsize=12, pad=10)
    ax.tick_params(colors=_TEXT, labelsize=8)
    plt.tight_layout()
    return fig


def plot_class_distribution(y, target_col="isFraud"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor=_DARK)
    for ax in axes:
        _dark_ax(ax)
    counts = y.value_counts().sort_index()
    labels = [f"Class {i}" for i in counts.index]
    colors = ["#43A047","#E53935"] if len(counts)==2 else _PALETTE[:len(counts)]
    axes[0].pie(counts.values, labels=labels, autopct="%1.2f%%",
                colors=colors, startangle=90, textprops={"color":"white"})
    axes[0].set_title(f"{target_col} Distribution", color="white")
    bars = axes[1].bar(labels, counts.values, color=colors)
    axes[1].set_title("Class Counts", color="white")
    for bar, v in zip(bars, counts.values):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     v + max(counts.values)*0.01,
                     f"{v:,}", ha="center", color="white", fontweight="bold", fontsize=9)
    plt.tight_layout()
    return fig


def plot_fraud_prob_histogram(proba, threshold=0.5):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_DARK)
    _dark_ax(ax, fig)
    ax.hist(proba, bins=50, color="#4A90D9", edgecolor=_DARK, alpha=0.85)
    ax.axvline(threshold, color="#E53935", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Fraud Probability", color=_TEXT)
    ax.set_ylabel("Transaction Count", color=_TEXT)
    ax.set_title("Fraud Probability Distribution", color="white", fontsize=12)
    ax.legend(facecolor=_CARD, labelcolor="white")
    plt.tight_layout()
    return fig
