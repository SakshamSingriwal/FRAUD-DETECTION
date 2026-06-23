"""
utils.py — Helper functions for the Fraud Detection Streamlit App.
Covers: feature engineering, preprocessing, model training, evaluation, prediction.
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
    RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

REQUIRED_COLS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"
]
TRAIN_COLS = REQUIRED_COLS + ["isFraud"]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations. Returns processed DataFrame."""
    df = df.copy()

    # Balance error features
    df["error_balance_orig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["error_balance_dest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    # Binary flags
    df["is_origin_emptied"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    ).astype(int)
    df["dest_initial_zero"] = (df["oldbalanceDest"] == 0).astype(int)
    df["is_merchant"] = df["nameDest"].str.startswith("M").astype(int)
    df["is_cash_out_transfer"] = df["type"].isin(
        ["CASH_OUT", "TRANSFER"]
    ).astype(int)

    # Log-transformed amount
    df["amount_log"] = np.log1p(df["amount"])

    # Balance change features
    df["orig_balance_change"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Ratio features (safe division)
    df["orig_balance_ratio"] = df["newbalanceOrig"] / (
        df["oldbalanceOrg"] + 1
    )
    df["dest_balance_ratio"] = df["newbalanceDest"] / (
        df["oldbalanceDest"] + 1
    )
    df["amount_to_orig_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # One-hot encode transaction type
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    # Drop identifier columns and raw type
    drop_cols = ["nameOrig", "nameDest", "step", "type"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


def remove_correlated_features(
    df: pd.DataFrame, threshold: float = 0.85, target_col: str = "isFraud"
) -> tuple[pd.DataFrame, list[str]]:
    """Drop numeric features with pairwise correlation above threshold."""
    feature_cols = [c for c in df.columns if c != target_col]
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df = df.drop(columns=to_drop)
    return df, to_drop


# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess(
    train_df: pd.DataFrame,
    test_size: float = 0.2,
    corr_threshold: float = 0.85,
    apply_smote: bool = True,
    random_state: int = 42
) -> dict:
    """
    Full preprocessing: feature engineering → correlation removal →
    train/test split → scaling → SMOTE.
    Returns a dict with all artefacts needed for training and prediction.
    """
    df = engineer_features(train_df)

    # Correlation removal
    df, dropped_corr = remove_correlated_features(
        df, threshold=corr_threshold
    )

    target = "isFraud"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if apply_smote and y_train.sum() > 0:
        sm = SMOTE(random_state=random_state)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    return {
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "dropped_corr": dropped_corr,
        "df_engineered": df,
    }


# ─────────────────────────────────────────────
# THRESHOLD TUNING (COST-SENSITIVE)
# ─────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 1.0,
    cost_fp: float = 0.1
) -> tuple[float, float]:
    """
    Find the threshold that minimises total misclassification cost.
    cost_fn = cost of a missed fraud (false negative)
    cost_fp = cost of a false alarm (false positive)
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh, best_cost = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_thresh = cost, t
    return best_thresh, best_cost


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def _get_proba(model, X) -> np.ndarray:
    """Return fraud probability scores, handling classifiers and anomaly detectors."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Isolation Forest: lower score = more anomalous → invert and normalise
        scores = -scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    return np.zeros(len(X))


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Compute all evaluation metrics and optimal threshold."""
    y_proba = _get_proba(model, X_test)
    threshold, min_cost = find_optimal_threshold(
        np.array(y_test), y_proba
    )
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Optimal Threshold": round(threshold, 4),
        "Min Cost": round(min_cost, 2),
        "y_proba": y_proba,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def build_model_registry(time_limit: int = 300) -> dict:
    """Return dict of model name → instantiated sklearn-compatible model."""
    registry = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100
        ),
    }

    # XGBoost
    try:
        from xgboost import XGBClassifier
        registry["XGBoost"] = XGBClassifier(
            n_estimators=100, scale_pos_weight=10,
            use_label_encoder=False, eval_metric="logloss",
            n_jobs=-1, verbosity=0
        )
    except ImportError:
        pass

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        registry["LightGBM"] = LGBMClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, verbose=-1
        )
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        registry["CatBoost"] = CatBoostClassifier(
            iterations=100, verbose=0, auto_class_weights="Balanced"
        )
    except ImportError:
        pass

    # Isolation Forest (anomaly)
    from sklearn.ensemble import IsolationForest
    registry["Isolation Forest"] = IsolationForest(
        contamination=0.01, n_jobs=-1, random_state=42
    )

    # Stacking Ensemble
    estimators = [
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
        ("dt", DecisionTreeClassifier(max_depth=6, class_weight="balanced")),
    ]
    try:
        from xgboost import XGBClassifier as _XGB
        estimators.append(
            ("xgb", _XGB(
                n_estimators=50, scale_pos_weight=10,
                use_label_encoder=False, eval_metric="logloss",
                verbosity=0
            ))
        )
    except ImportError:
        pass
    registry["Stacking Ensemble"] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced"),
        n_jobs=-1
    )

    return registry


def train_models(
    selected: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    time_limit: int = 300,
    progress_cb=None,
) -> dict:
    """
    Train selected models and return results dict.
    progress_cb(fraction, message) is called after each model completes.
    """
    registry = build_model_registry(time_limit)
    results = {}
    n = len(selected)

    for i, name in enumerate(selected):
        if name not in registry:
            continue
        model = registry[name]
        try:
            if name == "Isolation Forest":
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            metrics["model"] = model
            results[name] = metrics
        except Exception as e:
            results[name] = {"error": str(e)}

        if progress_cb:
            progress_cb((i + 1) / n, f"✅ Trained: {name}")

    return results


# ─────────────────────────────────────────────
# AUTOML WRAPPERS
# ─────────────────────────────────────────────

def train_h2o_automl(
    X_train, y_train, X_test, y_test, time_limit=300
) -> dict | None:
    try:
        import h2o
        from h2o.automl import H2OAutoML
        h2o.init(verbose=False)
        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train.values if hasattr(y_train, "values") else y_train
        h_train = h2o.H2OFrame(train_df)
        h_train["label"] = h_train["label"].asfactor()
        aml = H2OAutoML(max_runtime_secs=time_limit, seed=42, verbosity=None)
        aml.train(y="label", training_frame=h_train)
        test_df = pd.DataFrame(X_test)
        test_df["label"] = y_test.values if hasattr(y_test, "values") else y_test
        h_test = h2o.H2OFrame(test_df)
        h_test["label"] = h_test["label"].asfactor()
        preds = aml.leader.predict(h_test).as_data_frame()
        y_proba = preds["p1"].values
        threshold, min_cost = find_optimal_threshold(np.array(y_test), y_proba)
        y_pred = (y_proba >= threshold).astype(int)
        return {
            "model": aml.leader,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
            "Optimal Threshold": round(threshold, 4),
            "Min Cost": round(min_cost, 2),
            "y_proba": y_proba,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "is_h2o": True,
        }
    except Exception:
        return None


def train_autogluon(
    X_train, y_train, X_test, y_test, time_limit=300
) -> dict | None:
    try:
        from autogluon.tabular import TabularPredictor
        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train.values if hasattr(y_train, "values") else y_train
        predictor = TabularPredictor(
            label="label", eval_metric="roc_auc", verbosity=0
        ).fit(train_df, time_limit=time_limit)
        test_df = pd.DataFrame(X_test)
        y_proba = predictor.predict_proba(test_df).iloc[:, 1].values
        threshold, min_cost = find_optimal_threshold(np.array(y_test), y_proba)
        y_pred = (y_proba >= threshold).astype(int)
        return {
            "model": predictor,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
            "Optimal Threshold": round(threshold, 4),
            "Min Cost": round(min_cost, 2),
            "y_proba": y_proba,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "is_autogluon": True,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# SAVE / LOAD ARTEFACTS
# ─────────────────────────────────────────────

def save_artefacts(model, scaler, feature_cols, model_name):
    """Persist model, scaler, and feature columns to disk."""
    safe_name = model_name.replace(" ", "_")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(model_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))


def load_artefacts():
    """Load the saved best model, scaler, and feature columns."""
    name_path = os.path.join(MODELS_DIR, "best_model_name.pkl")
    if not os.path.exists(name_path):
        return None, None, None, None
    model_name = joblib.load(name_path)
    safe_name = model_name.replace(" ", "_")
    model = joblib.load(os.path.join(MODELS_DIR, f"{safe_name}.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    return model, scaler, feature_cols, model_name


# ─────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────

def prepare_input_for_prediction(
    raw_df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]
) -> np.ndarray:
    """Apply feature engineering + scaler to raw input data."""
    df = engineer_features(raw_df)
    # Align columns (fill missing engineered cols with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    return scaler.transform(df)


def predict_batch(
    raw_df: pd.DataFrame,
    model,
    scaler: StandardScaler,
    feature_cols: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return raw_df augmented with fraud_probability and fraud_prediction."""
    X = prepare_input_for_prediction(raw_df, scaler, feature_cols)
    proba = _get_proba(model, X)
    preds = (proba >= threshold).astype(int)
    out = raw_df.copy()
    out["fraud_probability"] = proba
    out["fraud_prediction"] = preds
    return out


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"], ax=ax
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    return fig


def plot_roc_curves(results: dict, y_test: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        if "y_proba" not in res:
            continue
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc = res.get("ROC-AUC", 0)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    return fig


def plot_pr_curves(results: dict, y_test: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        if "y_proba" not in res:
            continue
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ap = average_precision_score(y_test, res["y_proba"])
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="coolwarm",
        center=0, ax=ax, linewidths=0.3
    )
    ax.set_title("Correlation Heatmap (Lower Triangle)")
    plt.tight_layout()
    return fig


def plot_class_distribution(y: pd.Series) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    counts = y.value_counts()
    labels = ["Legitimate", "Fraud"]
    axes[0].pie(
        counts.values, labels=labels,
        autopct="%1.2f%%", colors=["#4CAF50", "#F44336"],
        startangle=90
    )
    axes[0].set_title("Class Distribution")
    axes[1].bar(labels, counts.values, color=["#4CAF50", "#F44336"])
    axes[1].set_title("Transaction Counts")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 50, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_fraud_prob_histogram(proba: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(proba, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", label="Threshold = 0.5")
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Distribution of Fraud Probabilities")
    ax.legend()
    plt.tight_layout()
    return fig
