"""
app.py — Main Streamlit entry point for the Fraud Detection App.
Run with: streamlit run app.py
"""

import io
import os
import warnings
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
    plot_fraud_prob_histogram, _get_proba,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "raw_df": None,
    "preprocessed": None,
    "model_results": {},
    "best_model_name": None,
    "best_model": None,
    "scaler": None,
    "feature_cols": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Detection Studio")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Data Upload & EDA",
        "🔧 Feature Engineering",
        "🤖 Model Training",
        "🎯 Single Prediction",
        "📁 Batch Prediction",
    ],
)

st.sidebar.markdown("---")
# Load previously saved artefacts
if st.sidebar.button("📂 Load Saved Model"):
    model, scaler, feature_cols, name = load_artefacts()
    if model:
        st.session_state.best_model = model
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols
        st.session_state.best_model_name = name
        st.sidebar.success(f"Loaded: **{name}**")
    else:
        st.sidebar.warning("No saved model found. Train one first.")

if st.session_state.best_model_name:
    st.sidebar.info(f"🏆 Active Model: **{st.session_state.best_model_name}**")


# ─────────────────────────────────────────────
# PAGE 1: DATA UPLOAD & EDA
# ─────────────────────────────────────────────
if page == "📊 Data Upload & EDA":
    st.title("📊 Data Upload & Exploratory Data Analysis")

    uploaded = st.file_uploader(
        "Upload Training CSV (must contain isFraud column)",
        type="csv",
        key="train_upload"
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        missing_cols = [c for c in TRAIN_COLS if c not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()
        st.session_state.raw_df = df
        st.success(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")

    df = st.session_state.raw_df
    if df is None:
        st.info("⬆️ Upload a training CSV file to begin.")
        st.stop()

    # ── Overview
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Fraud Cases", f"{df['isFraud'].sum():,}")
    c4.metric("Fraud Rate", f"{df['isFraud'].mean()*100:.2f}%")

    with st.expander("📄 Data Preview", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("📈 Basic Statistics"):
        st.dataframe(df.describe().T, use_container_width=True)

    with st.expander("🔎 Missing Values & Duplicates"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Missing Values:**")
            mv = df.isnull().sum()
            st.dataframe(mv[mv > 0].rename("count").to_frame(), use_container_width=True)
            if mv.sum() == 0:
                st.success("No missing values!")
        with c2:
            st.write(f"**Duplicate Rows:** {df.duplicated().sum():,}")

    # ── Class Distribution
    st.subheader("Class Distribution")
    fig = plot_class_distribution(df["isFraud"])
    st.pyplot(fig)
    plt.close()

    # ── Transaction Type Analysis
    st.subheader("Transaction Types")
    c1, c2 = st.columns(2)
    with c1:
        type_counts = df["type"].value_counts()
        st.dataframe(type_counts.rename("Count").to_frame(), use_container_width=True)
    with c2:
        fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
        fig2, ax = plt.subplots(figsize=(6, 3))
        fraud_by_type.plot(kind="bar", ax=ax, color="#E53935")
        ax.set_title("Fraud Rate by Transaction Type")
        ax.set_ylabel("Fraud Rate")
        ax.set_xlabel("")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Correlation Heatmap
    st.subheader("Correlation Heatmap (Numeric Features)")
    fig3 = plot_correlation_heatmap(df)
    st.pyplot(fig3)
    plt.close()


# ─────────────────────────────────────────────
# PAGE 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
elif page == "🔧 Feature Engineering":
    st.title("🔧 Feature Engineering & Preprocessing")

    df = st.session_state.raw_df
    if df is None:
        st.warning("⚠️ Upload a dataset on the Data Upload page first.")
        st.stop()

    st.markdown("### Configuration")
    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        corr_threshold = st.slider(
            "Correlation Removal Threshold", 0.5, 1.0, 0.85, 0.05
        )
    with c2:
        apply_smote = st.checkbox("Apply SMOTE to Training Set", value=True)

    if st.button("⚙️ Run Feature Engineering & Preprocessing"):
        with st.spinner("Processing…"):
            result = preprocess(
                df,
                test_size=test_size,
                corr_threshold=corr_threshold,
                apply_smote=apply_smote,
            )
        st.session_state.preprocessed = result
        st.success("✅ Preprocessing complete!")

    result = st.session_state.preprocessed
    if result is None:
        st.info("Click the button above to run preprocessing.")
        st.stop()

    # ── Summary
    st.subheader("Feature Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Features", len(result["feature_cols"]))
    c2.metric(
        "Training Samples (after SMOTE)",
        f"{len(result['X_train']):,}"
    )
    c3.metric("Test Samples", f"{len(result['X_test']):,}")

    if result["dropped_corr"]:
        st.warning(
            f"🗑️ Dropped {len(result['dropped_corr'])} correlated features: "
            f"`{', '.join(result['dropped_corr'])}`"
        )
    else:
        st.success("No features dropped by correlation filter.")

    with st.expander("📋 Final Feature Columns"):
        st.write(result["feature_cols"])

    with st.expander("🔬 Engineered Dataset Preview"):
        st.dataframe(
            result["df_engineered"].head(50), use_container_width=True
        )

    # Class balance after SMOTE
    y_train = result["y_train"]
    if hasattr(y_train, "value_counts"):
        counts = y_train.value_counts()
    else:
        unique, cnts = np.unique(y_train, return_counts=True)
        counts = pd.Series(cnts, index=unique)
    st.subheader("Training Set Class Balance (after SMOTE)")
    c1, c2 = st.columns(2)
    c1.metric("Legitimate", f"{counts.get(0, 0):,}")
    c2.metric("Fraud", f"{counts.get(1, 0):,}")


# ─────────────────────────────────────────────
# PAGE 3: MODEL TRAINING
# ─────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.title("🤖 Model Training & Comparison")

    result = st.session_state.preprocessed
    if result is None:
        st.warning("⚠️ Complete Feature Engineering first.")
        st.stop()

    # ── Model selection
    st.subheader("Select Models")
    all_models = list(build_model_registry().keys()) + ["H2O AutoML", "AutoGluon"]

    select_all = st.checkbox("Select All", value=False)
    cols = st.columns(3)
    selected = []
    for i, name in enumerate(all_models):
        default = select_all
        checked = cols[i % 3].checkbox(name, value=default, key=f"chk_{name}")
        if checked:
            selected.append(name)

    c1, c2 = st.columns(2)
    with c1:
        time_limit = st.slider(
            "AutoML Time Limit (seconds)", 60, 600, 300, 30
        )

    if not selected:
        st.info("Select at least one model to train.")
        st.stop()

    if st.button("🚀 Train Selected Models"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Separate AutoML models from standard
        standard = [m for m in selected if m not in ("H2O AutoML", "AutoGluon")]
        automl_models = [m for m in selected if m in ("H2O AutoML", "AutoGluon")]
        results = {}
        total = len(selected)
        counter = [0]  # mutable container to allow mutation inside nested function

        def progress_cb(frac, msg):
            counter[0] += 1
            progress_bar.progress(counter[0] / total)
            status_text.text(msg)

        if standard:
            r = train_models(
                standard,
                result["X_train"], result["X_test"],
                result["y_train"], result["y_test"],
                time_limit=time_limit,
                progress_cb=progress_cb,
            )
            results.update(r)

        for aml in automl_models:
            status_text.text(f"⏳ Running {aml}… (up to {time_limit}s)")
            if aml == "H2O AutoML":
                r = train_h2o_automl(
                    result["X_train"], result["y_train"],
                    result["X_test"], result["y_test"],
                    time_limit=time_limit,
                )
                if r:
                    results["H2O AutoML"] = r
                else:
                    st.warning("H2O AutoML not available or failed.")
            elif aml == "AutoGluon":
                r = train_autogluon(
                    result["X_train"], result["y_train"],
                    result["X_test"], result["y_test"],
                    time_limit=time_limit,
                )
                if r:
                    results["AutoGluon"] = r
                else:
                    st.warning("AutoGluon not available or failed.")
            counter[0] += 1
            progress_bar.progress(counter[0] / total)

        progress_bar.progress(1.0)
        status_text.text("✅ All training complete!")
        st.session_state.model_results = results

        # Find best by ROC-AUC
        best_name = max(
            (n for n, r in results.items() if "ROC-AUC" in r),
            key=lambda n: results[n]["ROC-AUC"],
            default=None
        )
        if best_name:
            st.session_state.best_model_name = best_name
            st.session_state.best_model = results[best_name]["model"]
            st.session_state.scaler = result["scaler"]
            st.session_state.feature_cols = result["feature_cols"]
            save_artefacts(
                results[best_name]["model"],
                result["scaler"],
                result["feature_cols"],
                best_name,
            )
            st.success(f"🏆 Best Model: **{best_name}** — saved to `models/`")

    results = st.session_state.model_results
    if not results:
        st.info("Train some models to see results here.")
        st.stop()

    y_test = result["y_test"]

    # ── Comparison Table
    st.subheader("📊 Model Comparison")
    metric_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
                   "Optimal Threshold", "Min Cost"]
    rows = []
    for name, res in results.items():
        if "error" in res:
            rows.append({"Model": name, "Error": res["error"]})
        else:
            row = {"Model": name}
            row.update({k: res.get(k, "—") for k in metric_keys})
            rows.append(row)

    compare_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    best = st.session_state.best_model_name

    def highlight_best(row):
        return [
            "background-color: #E8F5E9; font-weight: bold"
            if row["Model"] == best else ""
        ] * len(row)

    st.dataframe(
        compare_df.style.apply(highlight_best, axis=1),
        use_container_width=True
    )

    # ── Per-model metrics + confusion matrices
    st.subheader("🔎 Per-Model Details")
    for name, res in results.items():
        if "error" in res:
            st.error(f"{name}: {res['error']}")
            continue
        with st.expander(f"{'🏆 ' if name == best else ''}{name}"):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", res.get("Accuracy"))
            c2.metric("Precision", res.get("Precision"))
            c3.metric("Recall", res.get("Recall"))
            c4.metric("F1", res.get("F1"))
            c5.metric("ROC-AUC", res.get("ROC-AUC"))
            if "confusion_matrix" in res:
                fig = plot_confusion_matrix(
                    res["confusion_matrix"], title=f"{name} Confusion Matrix"
                )
                st.pyplot(fig)
                plt.close()

    # ── ROC & PR Curves
    valid_results = {
        n: r for n, r in results.items()
        if "y_proba" in r
    }
    if valid_results:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ROC Curves")
            fig = plot_roc_curves(valid_results, y_test)
            st.pyplot(fig)
            plt.close()
        with c2:
            st.subheader("Precision-Recall Curves")
            fig = plot_pr_curves(valid_results, y_test)
            st.pyplot(fig)
            plt.close()

    # ── Download best model
    if best and os.path.exists(f"models/{best.replace(' ', '_')}.pkl"):
        with open(f"models/{best.replace(' ', '_')}.pkl", "rb") as f:
            st.download_button(
                label=f"⬇️ Download Best Model ({best})",
                data=f,
                file_name=f"{best.replace(' ', '_')}.pkl",
                mime="application/octet-stream",
            )


# ─────────────────────────────────────────────
# PAGE 4: SINGLE TRANSACTION PREDICTION
# ─────────────────────────────────────────────
elif page == "🎯 Single Prediction":
    st.title("🎯 Single Transaction Fraud Prediction")

    model = st.session_state.best_model
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols

    if model is None:
        st.warning("⚠️ No model loaded. Train a model or load a saved one from the sidebar.")
        st.stop()

    # Model selector (if multiple trained)
    model_results = st.session_state.model_results
    if model_results:
        valid_names = [n for n, r in model_results.items() if "model" in r]
        chosen = st.selectbox(
            "Select Model for Prediction",
            valid_names,
            index=valid_names.index(st.session_state.best_model_name)
            if st.session_state.best_model_name in valid_names else 0
        )
        active_model = model_results[chosen]["model"]
        active_threshold = model_results[chosen].get("Optimal Threshold", 0.5)
    else:
        active_model = model
        active_threshold = 0.5

    st.markdown("### Enter Transaction Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        step = st.number_input("Step (hour)", min_value=1, value=1)
        t_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
    with c2:
        name_orig = st.text_input("nameOrig", value="C123456789")
        old_bal_orig = st.number_input("oldbalanceOrg", min_value=0.0, value=5000.0, format="%.2f")
        new_bal_orig = st.number_input("newbalanceOrig", min_value=0.0, value=4000.0, format="%.2f")
    with c3:
        name_dest = st.text_input("nameDest", value="C987654321")
        old_bal_dest = st.number_input("oldbalanceDest", min_value=0.0, value=0.0, format="%.2f")
        new_bal_dest = st.number_input("newbalanceDest", min_value=0.0, value=1000.0, format="%.2f")

    if st.button("🔍 Predict"):
        row = pd.DataFrame([{
            "step": step, "type": t_type, "amount": amount,
            "nameOrig": name_orig, "oldbalanceOrg": old_bal_orig,
            "newbalanceOrig": new_bal_orig, "nameDest": name_dest,
            "oldbalanceDest": old_bal_dest, "newbalanceDest": new_bal_dest,
        }])
        X = prepare_input_for_prediction(row, scaler, feature_cols)
        proba = _get_proba(active_model, X)[0]
        is_fraud = proba >= active_threshold

        st.markdown("---")
        if is_fraud:
            st.error(f"🚨 **FRAUD DETECTED** — Probability: `{proba:.4f}`")
        else:
            st.success(f"✅ **LEGITIMATE** — Fraud Probability: `{proba:.4f}`")

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Probability", f"{proba:.4f}")
        c2.metric("Threshold Used", f"{active_threshold:.4f}")
        c3.metric("Decision", "🚨 FRAUD" if is_fraud else "✅ LEGIT")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.barh(["Fraud Prob"], [proba], color="#F44336" if is_fraud else "#4CAF50")
        ax.barh(["Fraud Prob"], [1 - proba], left=[proba], color="#E0E0E0")
        ax.axvline(active_threshold, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_title(f"Fraud Probability: {proba:.2%}")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────
# PAGE 5: BATCH PREDICTION
# ─────────────────────────────────────────────
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction (Upload CSV)")

    model = st.session_state.best_model
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols

    if model is None:
        st.warning("⚠️ No model loaded. Train a model or load a saved one from the sidebar.")
        st.stop()

    active_threshold = 0.5
    if st.session_state.best_model_name and st.session_state.model_results:
        res = st.session_state.model_results.get(st.session_state.best_model_name, {})
        active_threshold = res.get("Optimal Threshold", 0.5)

    threshold_override = st.slider(
        "Decision Threshold", 0.01, 0.99, float(active_threshold), 0.01
    )

    uploaded = st.file_uploader(
        "Upload CSV for Batch Prediction", type="csv", key="batch_upload"
    )
    if not uploaded:
        st.info("Upload a CSV with transaction data.")
        st.stop()

    raw = pd.read_csv(uploaded)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    has_labels = "isFraud" in raw.columns
    st.success(f"✅ Loaded {len(raw):,} transactions")

    with st.spinner("Running predictions…"):
        output = predict_batch(raw, model, scaler, feature_cols, threshold_override)

    # ── Summary metrics
    st.subheader("Prediction Summary")
    n_fraud = output["fraud_prediction"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(output):,}")
    c2.metric("Predicted Fraud", f"{n_fraud:,}")
    c3.metric("Fraud Percentage", f"{n_fraud / len(output) * 100:.2f}%")

    # ── Histogram
    st.subheader("Fraud Probability Distribution")
    fig = plot_fraud_prob_histogram(output["fraud_probability"].values)
    st.pyplot(fig)
    plt.close()

    # ── Top suspicious
    st.subheader("🔴 Top 10 Most Suspicious Transactions")
    top10 = (
        output.sort_values("fraud_probability", ascending=False)
        .head(10)[["type", "amount", "nameOrig", "nameDest",
                    "fraud_probability", "fraud_prediction"]]
    )
    st.dataframe(top10, use_container_width=True)

    # ── Performance (if labels present)
    if has_labels:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        y_true = raw["isFraud"]
        y_pred = output["fraud_prediction"]
        y_prob = output["fraud_probability"]

        st.subheader("📈 Performance Against Ground Truth")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        c2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
        c3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
        c4.metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
        c5.metric("ROC-AUC", f"{roc_auc_score(y_true, y_prob):.4f}")

        from utils import plot_confusion_matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = plot_confusion_matrix(cm, "Batch Prediction Confusion Matrix")
        st.pyplot(fig)
        plt.close()

    # ── Download
    st.subheader("⬇️ Download Predictions")
    csv_bytes = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

    with st.expander("📄 Full Prediction Table"):
        st.dataframe(output, use_container_width=True)
