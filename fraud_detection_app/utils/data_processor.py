"""
data_processor.py — Auto-detection, feature engineering, and leakage-free
preprocessing.

Design notes (carried over from the production bug-fix work):
  * The decision threshold is later tuned on a VALIDATION split and scored on a
    held-out TEST split, so ``preprocess`` returns train/val/test.
  * Class imbalance is corrected exactly ONCE — SMOTE *or* class weights, never
    both. ``preprocess`` reports ``apply_smote`` so the trainer can decide.
  * Scaling, correlation removal, and SMOTE all learn from the TRAIN split only.
  * Everything is seeded with ``RANDOM_STATE`` for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from .constants import RANDOM_STATE

# PaySim schema — triggers the rich domain feature set when present.
PAYSIM_COLS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
               "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Columns that look like the fraud label, in priority order.
_TARGET_HINTS = ["isfraud", "is_fraud", "fraud", "class", "label",
                 "target", "is_anomaly", "anomaly"]
_ID_HINTS = ["id", "name", "uuid", "transactionid", "nameorig", "namedest"]


# ── Auto-detection ─────────────────────────────────────────────────────────────
def detect_metadata(df: pd.DataFrame) -> dict:
    """Inspect a dataframe and infer everything the app needs to drive itself."""
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    dt_cols      = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # Candidate binary targets: 2 unique non-null values.
    binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]

    target_col = _guess_target(df, binary_cols)
    problem_type = "supervised" if target_col else "unsupervised"

    fraud_count = fraud_rate = None
    if target_col is not None:
        pos = _positive_mask(df[target_col])
        fraud_count = int(pos.sum())
        fraud_rate = round(fraud_count / max(len(df), 1) * 100, 4)

    amount_col = _guess_amount(df, numeric_cols)

    return {
        "n_rows": n_rows, "n_cols": n_cols,
        "numeric_cols": numeric_cols, "categorical_cols": cat_cols,
        "datetime_cols": dt_cols, "binary_cols": binary_cols,
        "target_col": target_col, "problem_type": problem_type,
        "fraud_count": fraud_count, "fraud_rate": fraud_rate,
        "amount_col": amount_col,
        "missing_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "is_paysim": all(c in df.columns for c in PAYSIM_COLS),
    }


def _guess_target(df: pd.DataFrame, binary_cols: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for hint in _TARGET_HINTS:
        if hint in lower and lower[hint] in binary_cols:
            return lower[hint]
    # A near-name match (e.g. "isFraud" variants) that is binary.
    for c in binary_cols:
        if any(h in c.lower() for h in _TARGET_HINTS):
            return c
    return None


def _guess_amount(df: pd.DataFrame, numeric_cols: list[str]) -> str | None:
    lower = {c.lower(): c for c in numeric_cols}
    for key in ("amount", "amt", "value", "transactionamount"):
        if key in lower:
            return lower[key]
    return numeric_cols[0] if numeric_cols else None


def _positive_mask(series: pd.Series) -> pd.Series:
    """Treat 1 / True / 'fraud' / 'yes' as the positive (fraud) class."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) >= series.dropna().max()
    s = series.astype(str).str.lower()
    return s.isin(["1", "true", "fraud", "yes", "y", "anomaly", "positive"])


def normalize_target(series: pd.Series) -> pd.Series:
    """Map any 2-class target onto {0,1} with the rarer class as positive=1
    when string-typed, else the maximum value as positive."""
    return _positive_mask(series).astype(int)


def data_quality_report(df: pd.DataFrame) -> list[dict]:
    """Per-check health report with green/amber/red status."""
    rows = []
    n = max(len(df), 1)

    miss = df.isnull().sum().sum()
    miss_pct = miss / (n * max(df.shape[1], 1)) * 100
    rows.append(_check("Missing values", miss_pct, 1, 5,
                       f"{int(miss):,} cells ({miss_pct:.2f}%)"))

    dup_pct = df.duplicated().sum() / n * 100
    rows.append(_check("Duplicate rows", dup_pct, 1, 10,
                       f"{int(df.duplicated().sum()):,} rows ({dup_pct:.2f}%)"))

    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    rows.append(_check("Constant columns", len(const_cols), 0, 1,
                       f"{len(const_cols)} columns" if const_cols else "none"))

    rows.append({"check": "Row volume",
                 "status": "good" if n >= 1000 else ("warn" if n >= 100 else "bad"),
                 "detail": f"{n:,} rows"})
    return rows


def _check(name, value, good_max, warn_max, detail):
    status = "good" if value <= good_max else ("warn" if value <= warn_max else "bad")
    return {"check": name, "status": status, "detail": detail}


# ── Synthetic data ─────────────────────────────────────────────────────────────
def generate_synthetic_fraud(n_rows: int = 5000, fraud_rate: float = 0.03,
                             seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a PaySim-style dataset with realistic fraud patterns so users
    can try the app end-to-end without their own data."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    rows = []
    types = np.array(TRANSACTION_TYPES)

    for i in range(n_rows):
        is_fraud = i < n_fraud
        if is_fraud:
            # Fraud pattern: empty the origin account via TRANSFER / CASH_OUT.
            t = rng.choice(["TRANSFER", "CASH_OUT"])
            old_orig = rng.uniform(1_000, 200_000)
            amount = old_orig                      # drains the account
            new_orig = 0.0
            old_dest = 0.0
            new_dest = 0.0 if rng.random() < 0.7 else amount
        else:
            t = rng.choice(types)
            old_orig = rng.uniform(0, 150_000)
            amount = rng.uniform(1, max(old_orig, 1))
            new_orig = max(old_orig - amount, 0)
            old_dest = rng.uniform(0, 80_000)
            new_dest = old_dest + amount
        rows.append({
            "step": int(rng.integers(1, 744)), "type": t, "amount": round(amount, 2),
            "nameOrig": f"C{rng.integers(1e8, 1e9)}", "oldbalanceOrg": round(old_orig, 2),
            "newbalanceOrig": round(new_orig, 2),
            "nameDest": ("M" if rng.random() < 0.3 else "C") + str(rng.integers(1e8, 1e9)),
            "oldbalanceDest": round(old_dest, 2), "newbalanceDest": round(new_dest, 2),
            "isFraud": int(is_fraud),
        })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Feature engineering ─────────────────────────────────────────────────────────
def engineer_paysim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features for PaySim. Defensive against missing values so the same
    code path works for training and for single/batch prediction."""
    df = df.copy()
    for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)
    df["nameDest"] = df.get("nameDest", "").astype(str)
    df["type"] = df.get("type", "")

    df["error_balance_orig"]   = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["error_balance_dest"]   = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["is_origin_emptied"]    = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
    df["dest_initial_zero"]    = (df["oldbalanceDest"] == 0).astype(int)
    df["is_merchant"]          = df["nameDest"].str.startswith("M").astype(int)
    df["is_cash_out_transfer"] = df["type"].isin(["CASH_OUT", "TRANSFER"]).astype(int)
    df["amount_log"]           = np.log1p(df["amount"])
    df["orig_balance_change"]  = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_change"]  = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_orig_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)
    df.drop(columns=[c for c in ["nameOrig", "nameDest", "step", "type"] if c in df.columns],
            inplace=True)
    return df


def _encode_generic(df: pd.DataFrame, max_card: int = 20) -> pd.DataFrame:
    """One-hot encode low-cardinality categoricals; drop ID-like / high-cardinality
    columns to avoid feature explosion."""
    df = df.copy()
    drop, encode = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        nun = df[c].nunique(dropna=True)
        name_id = any(h in c.lower() for h in _ID_HINTS)
        if name_id or nun > max_card:
            drop.append(c)
        else:
            encode.append(c)
    df = df.drop(columns=drop)
    if encode:
        df = pd.get_dummies(df, columns=encode, drop_first=False)
    return df


_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


def build_feature_frame(df: pd.DataFrame, target_col: str | None,
                        feature_mode: str = "automatic",
                        manual_cols: list[str] | None = None) -> pd.DataFrame:
    """Return a numeric feature frame (no target) ready for splitting/scaling."""
    is_paysim = all(c in df.columns for c in PAYSIM_COLS)
    if feature_mode == "automatic" and is_paysim:
        feat = engineer_paysim_features(df.drop(columns=[target_col]) if target_col in df else df)
    else:
        if feature_mode == "manual" and manual_cols:
            cols = [c for c in manual_cols if c in df.columns and c != target_col]
        else:  # automatic on a non-PaySim dataset => use every non-target column
            cols = [c for c in df.columns if c != target_col]
        feat = _encode_generic(df[cols])
    # Coerce everything numeric, fill gaps (median per column, else 0).
    feat = feat.apply(pd.to_numeric, errors="coerce")
    feat = feat.fillna(feat.median(numeric_only=True)).fillna(0.0)
    return feat


def _find_correlated(X: pd.DataFrame, threshold: float) -> list[str]:
    if X.shape[1] < 2:
        return []
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if any(upper[c] > threshold)]


# ── Main preprocessing ──────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, target_col: str,
               test_size: float = 0.2, val_size: float = 0.2,
               scaler_kind: str = "standard", corr_threshold: float = 0.95,
               apply_corr_removal: bool = True, apply_smote: bool = True,
               feature_mode: str = "automatic", manual_cols: list[str] | None = None,
               random_state: int = RANDOM_STATE) -> dict:
    """Leakage-free train/val/test preprocessing for SUPERVISED training."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y_full = normalize_target(df[target_col])
    if y_full.nunique() < 2:
        raise ValueError(
            "Target has a single class — cannot train a supervised classifier. "
            "Switch to Unsupervised mode or check your label column."
        )

    X_full = build_feature_frame(df, target_col, feature_mode, manual_cols)
    feature_cols_all = X_full.columns.tolist()

    # 1. test split, 2. validation split (both stratified, both leakage-safe).
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full)
    strat = y_tr_full if y_tr_full.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=val_size, random_state=random_state, stratify=strat)

    # 3. correlation removal decided on TRAIN only.
    dropped = _find_correlated(X_train, corr_threshold) if apply_corr_removal else []
    feature_cols = [c for c in feature_cols_all if c not in dropped]
    X_train, X_val, X_test = X_train[feature_cols], X_val[feature_cols], X_test[feature_cols]

    # 4. scale (fit on TRAIN only).
    scaler = _SCALERS.get(scaler_kind, StandardScaler)()
    Xtr = scaler.fit_transform(X_train)
    Xvl = scaler.transform(X_val)
    Xte = scaler.transform(X_test)

    # 5. SMOTE on TRAIN only (skipped gracefully if it can't run).
    smote_applied = False
    if apply_smote and y_train.sum() >= 2:
        try:
            from imblearn.over_sampling import SMOTE
            k = min(5, int(y_train.sum()) - 1)
            Xtr, y_train = SMOTE(random_state=random_state, k_neighbors=max(k, 1)).fit_resample(Xtr, y_train)
            smote_applied = True
        except Exception:
            smote_applied = False

    return {
        "X_train": Xtr, "X_val": Xvl, "X_test": Xte,
        "y_train": np.asarray(y_train), "y_val": np.asarray(y_val), "y_test": np.asarray(y_test),
        "scaler": scaler, "feature_cols": feature_cols, "dropped_corr": dropped,
        "apply_smote": smote_applied, "target_col": target_col,
        "feature_mode": feature_mode, "manual_cols": manual_cols,
        "is_paysim": all(c in df.columns for c in PAYSIM_COLS),
        "class_balance": {int(k): int(v) for k, v in pd.Series(np.asarray(y_train)).value_counts().items()},
    }


def prepare_unsupervised(df: pd.DataFrame, scaler_kind: str = "standard",
                         feature_mode: str = "automatic",
                         manual_cols: list[str] | None = None,
                         target_col: str | None = None,
                         random_state: int = RANDOM_STATE) -> dict:
    """Scale the whole dataset for anomaly detection (no labels needed)."""
    X = build_feature_frame(df, target_col, feature_mode, manual_cols)
    scaler = _SCALERS.get(scaler_kind, StandardScaler)()
    Xs = scaler.fit_transform(X)
    return {"X": Xs, "scaler": scaler, "feature_cols": X.columns.tolist(),
            "feature_mode": feature_mode, "manual_cols": manual_cols}


def prepare_input_for_prediction(raw_df: pd.DataFrame, scaler, feature_cols: list[str],
                                 feature_mode: str = "automatic",
                                 manual_cols: list[str] | None = None,
                                 is_paysim: bool = True) -> np.ndarray:
    """Transform raw rows into the exact training feature matrix (same columns,
    order, dtype). Missing columns => 0, NaNs => 0, so prediction never crashes."""
    if feature_mode == "automatic" and is_paysim:
        df = engineer_paysim_features(raw_df)
    elif feature_mode == "manual" and manual_cols:
        df = _encode_generic(raw_df[[c for c in manual_cols if c in raw_df.columns]].copy())
    else:
        df = _encode_generic(raw_df.copy())
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return scaler.transform(X)
