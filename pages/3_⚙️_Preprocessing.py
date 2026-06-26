"""Page 3 — Preprocessing & feature engineering."""
import numpy as np
import pandas as pd
import streamlit as st

from utils.config import (setup_page, stat_card, explain, anchor, request_scroll,
                          apply_scroll)
from utils.data_processor import (preprocess, prepare_unsupervised, build_feature_frame,
                                  recommend_scaler)
from utils import visualizer as viz

setup_page("Preprocessing & Feature Engineering", "⚙️",
           "Configure the pipeline. Everything learns from the training split only — no leakage.",
           stage=2)

s = st.session_state
df = s.get("raw_df")
if df is None:
    st.warning("⚠️ Upload data first.")
    st.stop()

meta = s.get("meta") or {}
supervised = s.get("problem_type") == "supervised"

# ── Feature mode ────────────────────────────────────────────────────────────────
st.markdown("### 🧬 Feature selection")
mode_label = st.radio("Mode", ["🤖 Automatic", "✋ Manual", "🧠 Hybrid"], horizontal=True)
feature_mode = "automatic" if mode_label.startswith("🤖") else ("manual" if mode_label.startswith("✋") else "automatic")

manual_cols = None
if mode_label.startswith("✋") or mode_label.startswith("🧠"):
    candidates = [c for c in df.columns if c != s.get("target_col")]
    num_default = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    manual_cols = st.multiselect("Feature columns", candidates, default=num_default)
    if mode_label.startswith("🧠"):
        feature_mode = "manual"
    if not manual_cols:
        st.error("Select at least one feature column.")
        st.stop()

explain(
    "**Automatic** uses Sentinel's built-in fraud features when PaySim columns are "
    "present (balance-drain, error-balance, merchant flags…), otherwise every column. "
    "**Manual** lets you pick. **Hybrid** starts from your picks but lets auto-engineering "
    "enrich them. Categorical columns are one-hot encoded; ID-like / high-cardinality "
    "columns are dropped automatically.")

# ── Pipeline config ──────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Pipeline configuration")
c1, c2, c3 = st.columns(3)
with c1:
    scaler_choice = st.selectbox(
        "Scaler", ["Auto (recommended)", "standard", "robust", "minmax"],
        help="Auto profiles your data and picks the best scaler. Tree models (RF, "
             "XGBoost, CatBoost) are scale-invariant; scaling mainly helps Logistic "
             "Regression.")
    corr_thr = st.slider("Correlation removal threshold", 0.80, 1.0, 0.95, 0.01)
with c2:
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    val_size = st.slider("Validation size (of train)", 0.1, 0.4, 0.2, 0.05)
with c3:
    apply_corr = st.checkbox("Remove correlated features", value=True)
    apply_smote = st.checkbox("Balance with SMOTE", value=True, disabled=not supervised)

# Resolve the scaler: Auto = data-driven recommendation, else the manual pick.
if scaler_choice.startswith("Auto"):
    scaler_kind, scaler_reason = recommend_scaler(
        df, s.get("target_col"), feature_mode, manual_cols)
    st.caption(f"🤖 **Auto-selected scaler: `{scaler_kind}`** — {scaler_reason}")
else:
    scaler_kind = scaler_choice

if supervised:
    st.caption("ℹ️ When SMOTE is ON, models use **no** class weights (imbalance is "
               "corrected once). Turn SMOTE off to use class weights instead.")

# ── Run ──────────────────────────────────────────────────────────────────────────
if st.button("⚙️ Run preprocessing"):
    with st.spinner("Engineering features and splitting…"):
        try:
            if supervised:
                s.prep = preprocess(
                    df, s.target_col, test_size=test_size, val_size=val_size,
                    scaler_kind=scaler_kind, corr_threshold=corr_thr,
                    apply_corr_removal=apply_corr, apply_smote=apply_smote,
                    feature_mode=feature_mode, manual_cols=manual_cols)
            else:
                s.prep = prepare_unsupervised(
                    df, scaler_kind=scaler_kind, feature_mode=feature_mode,
                    manual_cols=manual_cols, target_col=s.target_col)
                s.prep["unsupervised"] = True
            s.scaler = s.prep["scaler"]
            s.feature_cols = s.prep["feature_cols"]
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()
    st.success("✅ Preprocessing complete.")
    request_scroll("prep-results")

prep = s.get("prep")
if not prep:
    st.info("Configure and run the pipeline to see results.")
    st.stop()

# ── Results ──────────────────────────────────────────────────────────────────────
anchor("prep-results")
st.markdown("### 📊 Pipeline results")
if prep.get("unsupervised"):
    a, b = st.columns(2)
    a.markdown(stat_card("Features", f"{len(prep['feature_cols'])}"), unsafe_allow_html=True)
    b.markdown(stat_card("Samples", f"{len(prep['X']):,}", "for anomaly detection"), unsafe_allow_html=True)
else:
    a, b, c, d = st.columns(4)
    a.markdown(stat_card("Features", f"{len(prep['feature_cols'])}"), unsafe_allow_html=True)
    b.markdown(stat_card("Train", f"{len(prep['X_train']):,}", "±SMOTE"), unsafe_allow_html=True)
    c.markdown(stat_card("Validation", f"{len(prep['X_val']):,}", "threshold tuning", tone="green"),
               unsafe_allow_html=True)
    d.markdown(stat_card("Test", f"{len(prep['X_test']):,}", "final metrics", tone="green"),
               unsafe_allow_html=True)
    st.caption("Threshold is tuned on validation; reported metrics use the untouched test split.")

    if prep["dropped_corr"]:
        st.warning(f"🗑️ Dropped {len(prep['dropped_corr'])} correlated features: "
                   f"`{', '.join(prep['dropped_corr'])}`")
    if prep.get("apply_smote"):
        bal = prep["class_balance"]
        st.markdown("**Class balance after SMOTE:**")
        st.plotly_chart(viz.class_distribution(
            np.r_[np.zeros(bal.get(0, 0)), np.ones(bal.get(1, 0))], "Training set"),
            width="stretch")

with st.expander("📋 Final feature columns"):
    st.dataframe(pd.DataFrame({"Feature": prep["feature_cols"]}), width="stretch")

apply_scroll()
