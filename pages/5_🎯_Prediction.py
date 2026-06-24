"""Page 5 — Prediction (supervised model OR unsupervised anomaly), single + batch."""
import numpy as np
import pandas as pd
import streamlit as st

from utils.config import setup_page, explain
from utils.data_processor import (prepare_input_for_prediction, PAYSIM_COLS,
                                  TRANSACTION_TYPES)
from utils.model_trainer import get_proba, load_artifacts
from utils.model_explainer import top_risk_factors, plain_english_prediction
from utils import unsupervised as un
from utils import visualizer as viz

setup_page("Prediction", "🎯",
           "Score single or batch transactions and see exactly why each was flagged.")

s = st.session_state

# Allow loading a previously saved model even in a fresh session.
with st.sidebar.expander("📂 Load saved model"):
    if st.button("Load from models/"):
        art = load_artifacts()
        if art:
            s.best_model = art["model"]; s.best_model_name = art["meta"]["name"]
            s.scaler = art["scaler"]; s.feature_cols = art["feature_cols"]
            s.prep = s.get("prep") or {"feature_mode": art["meta"].get("feature_mode", "automatic"),
                                       "is_paysim": art["meta"].get("is_paysim", True),
                                       "feature_cols": art["feature_cols"], "scaler": art["scaler"]}
            st.success(f"Loaded {s.best_model_name}")
        else:
            st.warning("No saved model found.")

prep = s.get("prep")
unsup = bool(prep and prep.get("unsupervised"))
has_supervised = s.get("best_model") is not None and not unsup

tabs = st.tabs(["🔹 Single prediction", "📦 Batch prediction"])

# ════════════════════════════════════════════════════════════════════════════════
# SINGLE PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if not (has_supervised or unsup):
        st.warning("⚠️ Train a model (or run anomaly detection) first.")
        st.stop()

    df_ref = s.get("raw_df")
    is_paysim = bool(prep.get("is_paysim")) if prep else (s.get("meta", {}) or {}).get("is_paysim", True)

    st.markdown("#### Enter transaction details")
    if is_paysim:
        c1, c2, c3 = st.columns(3)
        with c1:
            step = st.number_input("Step (hour)", 1, value=1)
            t_type = st.selectbox("Type", TRANSACTION_TYPES, index=4)
            amount = st.number_input("Amount", 0.0, value=181.0, format="%.2f")
        with c2:
            old_o = st.number_input("oldbalanceOrg", 0.0, value=181.0, format="%.2f")
            new_o = st.number_input("newbalanceOrig", 0.0, value=0.0, format="%.2f")
        with c3:
            old_d = st.number_input("oldbalanceDest", 0.0, value=0.0, format="%.2f")
            new_d = st.number_input("newbalanceDest", 0.0, value=0.0, format="%.2f")
        row = pd.DataFrame([{"step": step, "type": t_type, "amount": amount,
                             "nameOrig": "C1", "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
                             "nameDest": "C2", "oldbalanceDest": old_d, "newbalanceDest": new_d}])
    else:
        feats = (prep or {}).get("feature_cols", [])
        vals = {}
        cols = st.columns(min(3, max(1, len(feats))))
        # Use the source columns when available for sensible defaults.
        src = (prep or {}).get("manual_cols") or feats
        for i, f in enumerate(src):
            w = cols[i % len(cols)]
            if df_ref is not None and f in df_ref.columns and pd.api.types.is_numeric_dtype(df_ref[f]):
                vals[f] = w.number_input(f, value=float(df_ref[f].median()))
            elif df_ref is not None and f in df_ref.columns:
                vals[f] = w.selectbox(f, df_ref[f].dropna().unique().tolist())
            else:
                vals[f] = w.number_input(f, value=0.0)
        row = pd.DataFrame([vals])

    if st.button("🔍 Predict", key="single"):
        scaler = s.get("scaler"); feature_cols = s.get("feature_cols")
        X = prepare_input_for_prediction(row, scaler, feature_cols,
                                         feature_mode=(prep or {}).get("feature_mode", "automatic"),
                                         manual_cols=(prep or {}).get("manual_cols"),
                                         is_paysim=is_paysim)
        if unsup:
            # Score against a stored detector (rebuild on the fly from saved X).
            det = list(s.unsup_results.values())[0]
            # Risk via percentile of this row's distance — approximate with z-norm.
            expl = un.explain_anomaly(X[0], feature_cols)
            risk = float(np.clip(np.mean([e["severity"] for e in expl]) * 100, 0, 100))
            is_fraud = risk >= 50
            proba = risk / 100
            factors = None
        else:
            model = s.best_model
            proba = float(get_proba(model, X)[0])
            threshold = (prep or {}).get("threshold") or \
                s.get("results", {}).get(s.best_model_name, {}).get("Threshold", 0.5)
            is_fraud = proba >= threshold
            factors = top_risk_factors(model, X[0], feature_cols)

        cls = "verdict-fraud" if is_fraud else "verdict-legit"
        label = "🚨 FRAUD / ANOMALY" if is_fraud else "✅ LEGITIMATE"
        st.markdown(f"""<div class="verdict {cls}">
            <div class="verdict-label">{label}</div>
            <div class="verdict-sub">{'Risk' if unsup else 'Fraud probability'}: <b>{proba:.1%}</b></div>
            </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(viz.gauge(proba, "Risk" if unsup else "Fraud probability"),
                            width="stretch")
        with c2:
            if unsup:
                st.markdown("**Why this is unusual:**")
                st.markdown(un.anomaly_sentence(expl))
                for e in expl:
                    st.markdown(f"- `{e['feature']}` {e['direction']} (z={e['z']})")
            else:
                st.markdown(plain_english_prediction(proba, threshold, factors))
                st.plotly_chart(viz.waterfall_factors(factors), width="stretch")

# ════════════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if not (has_supervised or unsup):
        st.warning("⚠️ Train a model (or run anomaly detection) first.")
        st.stop()
    up = st.file_uploader("Upload CSV for scoring", type="csv", key="batch")
    if not up:
        st.info("Upload a CSV with the same schema as training.")
        st.stop()
    raw = pd.read_csv(up)
    is_paysim = bool(prep.get("is_paysim")) if prep else True

    scaler = s.get("scaler"); feature_cols = s.get("feature_cols")
    X = prepare_input_for_prediction(raw, scaler, feature_cols,
                                     feature_mode=(prep or {}).get("feature_mode", "automatic"),
                                     manual_cols=(prep or {}).get("manual_cols"), is_paysim=is_paysim)
    if unsup:
        name = list(s.unsup_results.keys())[0]
        res = un.run_detector(name, X, list(s.unsup_results.values())[0]["contamination"])
        proba = res["risk"] / 100
        threshold = 0.5
    else:
        proba = get_proba(s.best_model, X)
        threshold = (prep or {}).get("threshold") or \
            s.get("results", {}).get(s.best_model_name, {}).get("Threshold", 0.5)
    threshold = st.slider("Decision threshold", 0.01, 0.99, float(threshold), 0.01)
    out = raw.copy()
    out["fraud_probability"] = proba
    out["prediction"] = (proba >= threshold).astype(int)

    n_flag = int(out["prediction"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(out):,}")
    c2.metric("Flagged", f"{n_flag:,}")
    c3.metric("Flag rate", f"{n_flag/len(out)*100:.2f}%")
    st.plotly_chart(viz.probability_hist(proba, threshold), width="stretch")

    st.markdown("##### 🔴 Top 10 most suspicious")
    show = [c for c in ["type", "amount", "nameOrig", "nameDest"] if c in out.columns] + \
           ["fraud_probability", "prediction"]
    st.dataframe(out.sort_values("fraud_probability", ascending=False).head(10)[show],
                 width="stretch")

    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode("utf-8"),
                       "predictions.csv", "text/csv")
