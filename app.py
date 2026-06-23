"""
app.py — Fraud Detection Studio  v2.0
Run: streamlit run app.py
Features: selectable target column, automatic/manual feature mode, dark UI overhaul.
"""

import io, os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    REQUIRED_COLS, TRAIN_COLS,
    engineer_features, preprocess,
    build_model_registry, train_models,
    train_h2o_automl, train_autogluon,
    evaluate_model, find_optimal_threshold,
    save_artefacts, load_artefacts,
    prepare_input_for_prediction, predict_batch,
    plot_confusion_matrix, plot_roc_curves, plot_pr_curves,
    plot_correlation_heatmap, plot_class_distribution,
    plot_fraud_prob_histogram, get_proba,
    encode_categoricals, ModelNotPersistableError, AUTOML_MODEL_NAMES,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — dark card-based UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117 !important;
    color: #e5e7eb !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #12151e !important;
    border-right: 1px solid #2d3250;
}
/* ── Sidebar header ── */
.sidebar-logo {
    text-align: center;
    padding: 1.2rem 0.5rem 0.8rem;
    border-bottom: 1px solid #2d3250;
    margin-bottom: 1rem;
}
.sidebar-logo h2 { color: #4A90D9; margin: 0; font-size: 1.1rem; letter-spacing:.04em; }
.sidebar-logo p  { color: #6b7280; font-size: 0.72rem; margin: 2px 0 0; }
/* ── Nav pills ── */
[data-testid="stRadio"] > div { gap: 4px; }
[data-testid="stRadio"] label {
    background: #1e2130;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 0;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all .2s;
    color: #9ca3af !important;
    font-size: 0.85rem;
}
[data-testid="stRadio"] label:hover { border-color: #4A90D9; color: #e5e7eb !important; }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: #1a2744;
    border-color: #4A90D9;
    color: #4A90D9 !important;
}
/* ── Cards ── */
.card {
    background: #1e2130;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #2d3250;
    margin-bottom: 1rem;
}
.card-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #6b7280;
    margin-bottom: .3rem;
}
/* ── Page title ── */
.page-title {
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4A90D9 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.page-subtitle { color: #6b7280; font-size: 0.85rem; margin-bottom: 1.4rem; }
/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #2d3250 !important;
    border-radius: 12px !important;
    background: #12151e !important;
    padding: 1rem;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: #4A90D9 !important; }
/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 10px;
    border: 1px solid #2d3250;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricValue"] { color: #e5e7eb !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; }
/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4A90D9 0%, #3b73c2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: .02em;
    transition: opacity .2s, transform .15s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #2d3250; }
/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1e2130 !important;
    border-radius: 10px !important;
    border: 1px solid #2d3250 !important;
}
[data-testid="stExpander"] summary { color: #9ca3af !important; }
/* ── Selectbox / slider ── */
[data-baseweb="select"] > div { background: #1e2130 !important; border-color: #2d3250 !important; }
/* ── Result cards ── */
.result-fraud {
    background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
    border: 2px solid #E53935;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-legit {
    background: linear-gradient(135deg, #14532d 0%, #052e16 100%);
    border: 2px solid #43A047;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label { font-size: 2rem; font-weight: 800; letter-spacing: .06em; }
.result-prob  { font-size: 1rem; color: rgba(255,255,255,.75); margin-top:.4rem; }
/* ── Sidebar footer ── */
.sidebar-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 240px;
    padding: .8rem 1rem;
    background: #12151e;
    border-top: 1px solid #2d3250;
    font-size: .68rem;
    color: #4b5563;
    text-align: center;
}
/* ── Alert colors ── */
[data-testid="stAlert"] { border-radius: 10px !important; }
/* ── Config badge ── */
.config-badge {
    display: inline-block;
    background: #1a2744;
    color: #4A90D9;
    border: 1px solid #4A90D9;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "raw_df":         None,
    "preprocessed":   None,
    "model_results":  {},
    "best_model_name": None,
    "best_model":     None,
    "scaler":         None,
    "feature_cols":   None,
    # new config keys
    "target_col":     None,       # selected target column
    "feature_mode":   "Automatic",
    "manual_features": [],        # user-picked raw columns
    "feature_mode_locked": False, # True after preprocessing runs
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🛡️ Fraud Detection Studio</h2>
        <p>v2.0 &nbsp;·&nbsp; Production ML Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Data Upload & EDA",
         "🔧 Feature Engineering",
         "🤖 Model Training",
         "🎯 Single Prediction",
         "📁 Batch Prediction"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if st.button("📂 Load Saved Model"):
        out = load_artefacts()
        model, scaler, feature_cols, name, target, fmode = out
        if model:
            st.session_state.best_model      = model
            st.session_state.scaler          = scaler
            st.session_state.feature_cols    = feature_cols
            st.session_state.best_model_name = name
            st.session_state.target_col      = target
            st.session_state.feature_mode    = fmode.capitalize()
            st.success(f"Loaded: **{name}**")
        else:
            st.warning("No saved model found.")

    if st.session_state.best_model_name:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Active Model</div>
            <div style="color:#4A90D9;font-weight:700;">🏆 {st.session_state.best_model_name}</div>
            <div style="color:#6b7280;font-size:.75rem;margin-top:4px;">
                Target: <b>{st.session_state.target_col or '—'}</b> &nbsp;|&nbsp;
                Mode: <b>{st.session_state.feature_mode}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-footer">Fraud Detection Studio © 2025</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ── HELPER: page header
# ─────────────────────────────────────────────
def page_header(title, subtitle=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 1 — DATA UPLOAD & EDA
# ─────────────────────────────────────────────
if page == "📊 Data Upload & EDA":
    page_header("Data Upload & EDA",
                "Upload a CSV, pick your target column, and explore the dataset.")

    # ── Upload
    uploaded = st.file_uploader(
        "Drop your training CSV here", type="csv", key="train_upload",
        help="The file should contain a binary target column (0/1 or True/False)."
    )
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.raw_df = df
        # Reset downstream state when a new file is loaded
        st.session_state.preprocessed      = None
        st.session_state.model_results     = {}
        st.session_state.best_model        = None
        st.session_state.feature_cols      = None
        st.session_state.feature_mode_locked = False
        st.success(f"✅ Loaded **{len(df):,}** rows × **{df.shape[1]}** columns")

    df = st.session_state.raw_df
    if df is None:
        st.info("⬆️ Upload a training CSV file to begin.")
        st.stop()

    # ── Target column selection ──────────────────────────────────────────────
    st.markdown("### ⚙️ Dataset Configuration")

    all_cols    = df.columns.tolist()
    binary_cols = [c for c in all_cols
                   if df[c].nunique() == 2 and pd.api.types.is_numeric_dtype(df[c])]
    default_target = "isFraud" if "isFraud" in all_cols else (binary_cols[0] if binary_cols else all_cols[-1])

    cfg_c1, cfg_c2 = st.columns([1, 2])
    with cfg_c1:
        target_col = st.selectbox(
            "🎯 Target Column",
            all_cols,
            index=all_cols.index(default_target),
            help="Select the binary column you want to predict (0 = legitimate, 1 = fraud/positive).",
        )
    with cfg_c2:
        feature_mode = st.radio(
            "🔬 Feature Selection Mode",
            ["Automatic", "Manual"],
            horizontal=True,
            help=(
                "**Automatic**: uses domain-specific engineered features (recommended for PaySim data).\n"
                "**Manual**: you select which raw columns to use as features."
            ),
        )

    st.session_state.target_col   = target_col
    st.session_state.feature_mode = feature_mode

    if feature_mode == "Manual":
        raw_feature_candidates = [c for c in all_cols if c != target_col]
        # Smart default: pre-select numeric columns
        numeric_candidates = [c for c in raw_feature_candidates
                              if pd.api.types.is_numeric_dtype(df[c])]
        default_manual = st.session_state.manual_features or numeric_candidates
        default_manual = [c for c in default_manual if c in raw_feature_candidates]

        manual_features = st.multiselect(
            "📋 Select Features",
            raw_feature_candidates,
            default=default_manual,
            help="Pick the raw columns to use as model inputs. Categorical columns will be one-hot encoded automatically.",
        )
        st.session_state.manual_features = manual_features

        cat_in_selection = [c for c in manual_features
                            if not pd.api.types.is_numeric_dtype(df[c])]
        if cat_in_selection:
            st.info(f"ℹ️ Categorical columns will be one-hot encoded: **{', '.join(cat_in_selection)}**")
    else:
        # Show which PaySim columns the automatic mode needs
        paysim_cols = ["step","type","amount","nameOrig","oldbalanceOrg",
                       "newbalanceOrig","nameDest","oldbalanceDest","newbalanceDest"]
        missing_paysim = [c for c in paysim_cols if c not in df.columns]
        if missing_paysim:
            st.warning(
                f"⚠️ Automatic mode expects PaySim columns. Missing: `{', '.join(missing_paysim)}`. "
                "Switch to **Manual** mode or upload a compatible dataset."
            )
        else:
            st.success("✅ All PaySim columns detected — Automatic feature engineering ready.")

    st.divider()

    # ── Summary metrics
    st.markdown("### 📊 Dataset Overview")
    target_counts = df[target_col].value_counts().sort_index()
    fraud_count   = int(target_counts.get(1, 0))
    fraud_rate    = fraud_count / len(df) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",        f"{len(df):,}")
    c2.metric("Columns",     df.shape[1])
    c3.metric("Positive Cases", f"{fraud_count:,}")
    c4.metric("Positive Rate",  f"{fraud_rate:.2f}%")

    with st.expander("📄 Data Preview", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("📈 Basic Statistics"):
        st.dataframe(df.describe().T, use_container_width=True)

    with st.expander("🔎 Missing Values & Duplicates"):
        c1, c2 = st.columns(2)
        with c1:
            mv = df.isnull().sum()
            if mv.sum() == 0:
                st.success("No missing values!")
            else:
                st.dataframe(mv[mv > 0].rename("count").to_frame(), use_container_width=True)
        with c2:
            st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    # ── Class distribution
    st.markdown("### Class Distribution")
    fig = plot_class_distribution(df[target_col], target_col)
    st.pyplot(fig); plt.close()

    # ── Transaction types (only if 'type' column exists)
    if "type" in df.columns:
        st.markdown("### Transaction Types")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(df["type"].value_counts().rename("Count").to_frame(),
                         use_container_width=True)
        with c2:
            fraud_by_type = df.groupby("type")[target_col].mean().sort_values(ascending=False)
            fig2, ax = plt.subplots(figsize=(6, 3), facecolor="#0f1117")
            ax.set_facecolor("#0f1117")
            bars = ax.bar(fraud_by_type.index, fraud_by_type.values, color="#E53935")
            ax.set_title(f"Positive Rate by Transaction Type", color="white")
            ax.set_ylabel("Rate", color="#9ca3af")
            ax.tick_params(colors="#9ca3af")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3250")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig2); plt.close()

    # ── Correlation heatmap
    st.markdown("### Correlation Heatmap")
    fig3 = plot_correlation_heatmap(df)
    st.pyplot(fig3); plt.close()


# ─────────────────────────────────────────────
# PAGE 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
elif page == "🔧 Feature Engineering":
    page_header("Feature Engineering & Preprocessing",
                "Configure your preprocessing pipeline and run it.")

    df         = st.session_state.raw_df
    target_col = st.session_state.target_col
    mode       = st.session_state.feature_mode

    if df is None:
        st.warning("⚠️ Upload a dataset on the Data Upload page first.")
        st.stop()
    if target_col is None:
        st.warning("⚠️ Select a target column on the Data Upload page first.")
        st.stop()

    # Current config summary
    badges = f"""
    <span class="config-badge">Target: {target_col}</span>
    <span class="config-badge">Mode: {mode}</span>
    """
    if mode == "Manual":
        mf = st.session_state.manual_features
        badges += f'<span class="config-badge">{len(mf)} features selected</span>'
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown("")

    st.markdown("### ⚙️ Pipeline Configuration")
    c1, c2 = st.columns(2)
    with c1:
        test_size      = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        corr_threshold = st.slider("Correlation Removal Threshold", 0.5, 1.0, 0.85, 0.05)
    with c2:
        apply_smote       = st.checkbox("Apply SMOTE to Training Set", value=True)
        apply_corr_removal = st.checkbox("Apply Correlation Removal", value=True)

    if mode == "Manual" and not st.session_state.manual_features:
        st.error("⚠️ No features selected. Go to Data Upload → select features in Manual mode.")
        st.stop()

    if st.button("⚙️ Run Preprocessing Pipeline", use_container_width=False):
        with st.spinner("Engineering features and preprocessing…"):
            result = preprocess(
                df,
                test_size=test_size,
                corr_threshold=corr_threshold,
                apply_smote=apply_smote,
                target_col=target_col,
                feature_mode=mode.lower(),
                manual_feature_cols=st.session_state.manual_features,
                apply_corr_removal=apply_corr_removal,
            )
        st.session_state.preprocessed       = result
        st.session_state.feature_mode_locked = True
        st.success("✅ Preprocessing complete!")

    result = st.session_state.preprocessed
    if result is None:
        st.info("Click the button above to run the pipeline.")
        st.stop()

    # ── Results summary
    st.markdown("### 📊 Pipeline Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features",               len(result["feature_cols"]))
    c2.metric("Train Samples (±SMOTE)", f"{len(result['X_train']):,}")
    c3.metric("Validation Samples",     f"{len(result['X_val']):,}")
    c4.metric("Test Samples",           f"{len(result['X_test']):,}")
    st.caption("Threshold is tuned on the validation split; reported metrics use the untouched test split.")

    if result["dropped_corr"]:
        st.warning(f"🗑️ Dropped **{len(result['dropped_corr'])}** correlated features: "
                   f"`{', '.join(result['dropped_corr'])}`")
    else:
        st.success("No features removed by correlation filter.")

    with st.expander("📋 Final Feature Columns"):
        cols_df = pd.DataFrame({"Feature": result["feature_cols"]})
        st.dataframe(cols_df, use_container_width=True)

    with st.expander("🔬 Engineered Dataset Preview"):
        st.dataframe(result["df_engineered"].head(50), use_container_width=True)

    # Class balance after SMOTE
    y_train = result["y_train"]
    if hasattr(y_train, "value_counts"):
        counts = y_train.value_counts()
    else:
        unique, cnts = np.unique(y_train, return_counts=True)
        counts = pd.Series(cnts, index=unique)

    st.markdown("### Training Set Class Balance (after SMOTE)")
    c1, c2 = st.columns(2)
    c1.metric("Class 0 (Negative)", f"{counts.get(0, 0):,}")
    c2.metric("Class 1 (Positive)", f"{counts.get(1, 0):,}")


# ─────────────────────────────────────────────
# PAGE 3 — MODEL TRAINING
# ─────────────────────────────────────────────
elif page == "🤖 Model Training":
    page_header("Model Training & Comparison",
                "Select models, train, and compare performance metrics.")

    result = st.session_state.preprocessed
    if result is None:
        st.warning("⚠️ Run Feature Engineering first.")
        st.stop()

    target_col = result.get("target_col", st.session_state.target_col or "isFraud")
    mode       = result.get("feature_mode", "automatic")
    badge_html = f"""
    <span class="config-badge">Target: {target_col}</span>
    <span class="config-badge">Mode: {mode.capitalize()}</span>
    <span class="config-badge">{len(result['feature_cols'])} features</span>
    """
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("")

    # ── Model selection
    st.markdown("### 🤖 Select Models to Train")
    all_models = list(build_model_registry().keys()) + ["H2O AutoML", "AutoGluon"]

    select_all = st.checkbox("Select All", value=False)
    cols       = st.columns(3)
    selected   = []
    for i, name in enumerate(all_models):
        checked = cols[i % 3].checkbox(name, value=select_all, key=f"chk_{name}")
        if checked:
            selected.append(name)

    c1, _ = st.columns([1, 2])
    with c1:
        time_limit = st.slider("AutoML Time Limit (s)", 60, 600, 300, 30)

    if not selected:
        st.info("Select at least one model to continue.")
        st.stop()

    if st.button("🚀 Train Selected Models", use_container_width=False):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        standard     = [m for m in selected if m not in ("H2O AutoML", "AutoGluon")]
        automl_list  = [m for m in selected if m in ("H2O AutoML", "AutoGluon")]
        results      = {}
        total        = len(selected)
        counter      = [0]

        def progress_cb(frac, msg):
            counter[0] += 1
            progress_bar.progress(min(counter[0] / total, 1.0))
            status_text.text(msg)

        # SMOTE and class weights must not both correct the imbalance.
        # Only fall back to class weights when SMOTE was NOT applied.
        use_class_weights = not result.get("apply_smote", False)

        if standard:
            r = train_models(
                standard,
                result["X_train"], result["y_train"],
                result["X_val"], result["y_val"],
                result["X_test"], result["y_test"],
                use_class_weights=use_class_weights,
                time_limit=time_limit, progress_cb=progress_cb,
            )
            results.update(r)

        for aml in automl_list:
            status_text.text(f"⏳ Running {aml}… (up to {time_limit}s)")
            if aml == "H2O AutoML":
                r = train_h2o_automl(result["X_train"], result["y_train"],
                                     result["X_val"], result["y_val"],
                                     result["X_test"], result["y_test"], time_limit)
                if r: results["H2O AutoML"] = r
                else: st.warning("H2O AutoML not available or failed.")
            elif aml == "AutoGluon":
                r = train_autogluon(result["X_train"], result["y_train"],
                                    result["X_val"], result["y_val"],
                                    result["X_test"], result["y_test"], time_limit)
                if r: results["AutoGluon"] = r
                else: st.warning("AutoGluon not available or failed.")
            counter[0] += 1
            progress_bar.progress(min(counter[0] / total, 1.0))

        progress_bar.progress(1.0)
        status_text.text("✅ All training complete!")
        st.session_state.model_results = results

        best_name = max(
            (n for n, r in results.items() if "ROC-AUC" in r),
            key=lambda n: results[n]["ROC-AUC"],
            default=None,
        )
        if best_name:
            st.session_state.best_model_name = best_name
            st.session_state.best_model      = results[best_name]["model"]
            st.session_state.scaler          = result["scaler"]
            st.session_state.feature_cols    = result["feature_cols"]
            try:
                save_artefacts(
                    results[best_name]["model"],
                    result["scaler"],
                    result["feature_cols"],
                    best_name,
                    target_col=target_col,
                    feature_mode=mode,
                )
                st.success(f"🏆 Best Model: **{best_name}** — saved to `models/`")
            except ModelNotPersistableError as e:
                # AutoML leaders can't be pickled — keep them for this session only.
                st.warning(f"🏆 Best Model: **{best_name}**. {e}")

    results = st.session_state.model_results
    if not results:
        st.info("Train models to see results here.")
        st.stop()

    y_test = result["y_test"]
    best   = st.session_state.best_model_name

    # ── Comparison table
    st.markdown("### 📊 Model Comparison")
    metric_keys = ["Accuracy","Precision","Recall","F1","ROC-AUC","Optimal Threshold","Min Cost"]
    rows = []
    for name, res in results.items():
        if "error" in res:
            rows.append({"Model": name, "Error": res["error"]})
        else:
            row = {"Model": name}
            row.update({k: res.get(k, "—") for k in metric_keys})
            rows.append(row)

    compare_df = pd.DataFrame(rows)
    if "ROC-AUC" in compare_df.columns:
        compare_df = compare_df.sort_values("ROC-AUC", ascending=False)

    def highlight_best(row):
        style = "background-color: #1a2744; color: #4A90D9; font-weight: bold" \
                if row["Model"] == best else ""
        return [style] * len(row)

    st.dataframe(
        compare_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
    )

    # ── Per-model expanders
    st.markdown("### 🔎 Per-Model Details")
    for name, res in results.items():
        if "error" in res:
            st.error(f"{name}: {res['error']}")
            continue
        crown = "🏆 " if name == best else ""
        with st.expander(f"{crown}{name}"):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  res.get("Accuracy"))
            c2.metric("Precision", res.get("Precision"))
            c3.metric("Recall",    res.get("Recall"))
            c4.metric("F1",        res.get("F1"))
            c5.metric("ROC-AUC",   res.get("ROC-AUC"))
            if "confusion_matrix" in res:
                fig = plot_confusion_matrix(res["confusion_matrix"], f"{name} Confusion Matrix")
                st.pyplot(fig); plt.close()

    # ── ROC & PR curves
    valid = {n: r for n, r in results.items() if "y_proba" in r}
    if valid:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ROC Curves**")
            fig = plot_roc_curves(valid, y_test)
            st.pyplot(fig); plt.close()
        with c2:
            st.markdown("**Precision-Recall Curves**")
            fig = plot_pr_curves(valid, y_test)
            st.pyplot(fig); plt.close()

    # ── Download best model
    if best:
        safe = best.replace(" ", "_")
        path = f"models/{safe}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download Best Model ({best})",
                    data=f,
                    file_name=f"{safe}.pkl",
                    mime="application/octet-stream",
                )


# ─────────────────────────────────────────────
# PAGE 4 — SINGLE PREDICTION
# ─────────────────────────────────────────────
elif page == "🎯 Single Prediction":
    page_header("Single Transaction Prediction",
                "Enter transaction details and get an instant fraud assessment.")

    model        = st.session_state.best_model
    scaler       = st.session_state.scaler
    feature_cols = st.session_state.feature_cols
    target_col   = st.session_state.target_col or "isFraud"
    mode         = st.session_state.feature_mode.lower() if st.session_state.feature_mode else "automatic"

    if model is None:
        st.warning("⚠️ No model loaded. Train a model or load a saved one from the sidebar.")
        st.stop()

    # Model selector
    model_results = st.session_state.model_results
    if model_results:
        valid_names = [n for n, r in model_results.items() if "model" in r]
        default_idx = valid_names.index(st.session_state.best_model_name) \
                      if st.session_state.best_model_name in valid_names else 0
        chosen       = st.selectbox("Select Model for Prediction", valid_names, index=default_idx)
        active_model = model_results[chosen]["model"]
        active_threshold = model_results[chosen].get("Optimal Threshold", 0.5)
    else:
        active_model     = model
        active_threshold = 0.5
        chosen           = st.session_state.best_model_name or "Loaded Model"

    st.markdown(f"**Using:** `{chosen}` &nbsp;·&nbsp; **Threshold:** `{active_threshold:.4f}`")
    st.divider()

    # ── Input form — depends on feature mode
    if mode == "automatic":
        # PaySim-style form
        st.markdown("### Enter Transaction Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            step       = st.number_input("Step (hour)", min_value=1, value=1)
            t_type     = st.selectbox("Type", ["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT"])
            amount     = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
        with c2:
            name_orig   = st.text_input("nameOrig",      value="C123456789")
            old_bal_orig = st.number_input("oldbalanceOrg",  min_value=0.0, value=5000.0, format="%.2f")
            new_bal_orig = st.number_input("newbalanceOrig", min_value=0.0, value=4000.0, format="%.2f")
        with c3:
            name_dest   = st.text_input("nameDest",       value="C987654321")
            old_bal_dest = st.number_input("oldbalanceDest",  min_value=0.0, value=0.0,    format="%.2f")
            new_bal_dest = st.number_input("newbalanceDest",  min_value=0.0, value=1000.0, format="%.2f")

        row = pd.DataFrame([{
            "step": step, "type": t_type, "amount": amount,
            "nameOrig": name_orig, "oldbalanceOrg": old_bal_orig,
            "newbalanceOrig": new_bal_orig, "nameDest": name_dest,
            "oldbalanceDest": old_bal_dest, "newbalanceDest": new_bal_dest,
        }])

    else:  # manual mode — show inputs for selected raw features
        manual_features = st.session_state.manual_features or []
        df_ref   = st.session_state.raw_df
        st.markdown("### Enter Feature Values")
        input_vals = {}
        cols = st.columns(min(3, max(1, len(manual_features))))
        for i, feat in enumerate(manual_features):
            col_widget = cols[i % len(cols)]
            if df_ref is not None and feat in df_ref.columns:
                if pd.api.types.is_numeric_dtype(df_ref[feat]):
                    input_vals[feat] = col_widget.number_input(feat, value=float(df_ref[feat].median()))
                else:
                    unique_vals = df_ref[feat].dropna().unique().tolist()
                    input_vals[feat] = col_widget.selectbox(feat, unique_vals)
            else:
                input_vals[feat] = col_widget.text_input(feat, value="0")
        row = pd.DataFrame([input_vals])

    if st.button("🔍 Predict"):
        manual_raw = st.session_state.manual_features if mode == "manual" else None
        X          = prepare_input_for_prediction(row, scaler, feature_cols, mode, manual_raw)
        proba      = float(get_proba(active_model, X)[0])
        is_fraud   = proba >= active_threshold

        st.markdown("---")
        if is_fraud:
            st.markdown(f"""
            <div class="result-fraud">
                <div class="result-label">🚨 FRAUD DETECTED</div>
                <div class="result-prob">Fraud Probability: <b>{proba:.4f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
                <div class="result-label">✅ LEGITIMATE</div>
                <div class="result-prob">Fraud Probability: <b>{proba:.4f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Probability", f"{proba:.4f}")
        c2.metric("Threshold Used",    f"{active_threshold:.4f}")
        c3.metric("Decision",          "🚨 FRAUD" if is_fraud else "✅ LEGIT")

        # Gauge bar
        fig, ax = plt.subplots(figsize=(5, 1.2), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        ax.barh([""], [proba],       color="#E53935" if is_fraud else "#43A047", height=0.5)
        ax.barh([""], [1 - proba],   left=[proba], color="#2d3250",             height=0.5)
        ax.axvline(active_threshold, color="white", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_title(f"Fraud Probability: {proba:.2%}", color="white", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────
# PAGE 5 — BATCH PREDICTION
# ─────────────────────────────────────────────
elif page == "📁 Batch Prediction":
    page_header("Batch Prediction",
                "Upload a transaction CSV and get fraud scores for every row.")

    model        = st.session_state.best_model
    scaler       = st.session_state.scaler
    feature_cols = st.session_state.feature_cols
    target_col   = st.session_state.target_col or "isFraud"
    mode         = st.session_state.feature_mode.lower() if st.session_state.feature_mode else "automatic"
    manual_raw   = st.session_state.manual_features if mode == "manual" else None

    if model is None:
        st.warning("⚠️ No model loaded. Train a model or load a saved one from the sidebar.")
        st.stop()

    # Default threshold
    active_threshold = 0.5
    if st.session_state.best_model_name and st.session_state.model_results:
        res = st.session_state.model_results.get(st.session_state.best_model_name, {})
        active_threshold = res.get("Optimal Threshold", 0.5)

    threshold_override = st.slider(
        "Decision Threshold", 0.01, 0.99, float(active_threshold), 0.01
    )

    uploaded = st.file_uploader("Upload CSV for Batch Prediction", type="csv", key="batch_upload")
    if not uploaded:
        st.info("Upload a CSV with the same schema as your training data.")
        st.stop()

    raw = pd.read_csv(uploaded)
    if mode == "automatic":
        missing = [c for c in REQUIRED_COLS if c not in raw.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            st.stop()
    else:
        if manual_raw:
            missing = [c for c in manual_raw if c not in raw.columns]
            if missing:
                st.error(f"❌ Missing selected feature columns: {missing}")
                st.stop()

    has_labels = target_col in raw.columns
    st.success(f"✅ Loaded **{len(raw):,}** transactions")

    with st.spinner("Running predictions…"):
        output = predict_batch(
            raw, model, scaler, feature_cols,
            threshold=threshold_override,
            feature_mode=mode,
            manual_raw_cols=manual_raw,
        )

    # ── Summary metrics
    st.markdown("### 📊 Prediction Summary")
    n_fraud = int(output["fraud_prediction"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(output):,}")
    c2.metric("Predicted Fraud",    f"{n_fraud:,}")
    c3.metric("Fraud Percentage",   f"{n_fraud / len(output) * 100:.2f}%")

    # ── Histogram
    st.markdown("### Fraud Probability Distribution")
    fig = plot_fraud_prob_histogram(output["fraud_probability"].values, threshold_override)
    st.pyplot(fig); plt.close()

    # ── Top suspicious
    st.markdown("### 🔴 Top 10 Most Suspicious Transactions")
    common_display = ["fraud_probability", "fraud_prediction"]
    display_cols   = [c for c in ["type","amount","nameOrig","nameDest"] if c in output.columns]
    display_cols  += common_display
    top10 = output.sort_values("fraud_probability", ascending=False).head(10)[display_cols]
    st.dataframe(top10, use_container_width=True)

    # ── Live performance (if labels present)
    if has_labels:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )
        y_true = raw[target_col]
        y_pred = output["fraud_prediction"]
        y_prob = output["fraud_probability"]

        st.markdown("### 📈 Performance Against Ground Truth")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.4f}")
        c2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
        c3.metric("Recall",    f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
        c4.metric("F1",        f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
        c5.metric("ROC-AUC",   f"{roc_auc_score(y_true, y_prob):.4f}")

        from sklearn.metrics import confusion_matrix as cm_fn
        cm  = cm_fn(y_true, y_pred)
        fig = plot_confusion_matrix(cm, "Batch Prediction Confusion Matrix")
        st.pyplot(fig); plt.close()

    # ── Download
    st.markdown("### ⬇️ Download Predictions")
    csv_bytes = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

    with st.expander("📄 Full Prediction Table"):
        st.dataframe(output, use_container_width=True)