"""Page 6 — Model Explainability (SHAP, global importance, what-if analysis)."""
import numpy as np
import streamlit as st

from utils.config import setup_page, explain
from utils.model_trainer import get_proba
from utils.model_explainer import (global_importance, shap_values_for,
                                   shap_available, top_risk_factors,
                                   plain_english_prediction)
from utils import visualizer as viz

setup_page("Model Explainability", "📚",
           "Understand what the model learned — globally and for any single transaction.")

s = st.session_state
prep = s.get("prep")
model = s.get("best_model")
if model is None or prep is None or prep.get("unsupervised"):
    st.warning("⚠️ Train a supervised model first (Explainability needs a trained classifier).")
    st.stop()

feature_cols = prep["feature_cols"]
X_test = prep["X_test"]

if not shap_available():
    st.info("ℹ️ SHAP is not installed — showing native / permutation importance instead. "
            "`pip install shap` for richer explanations.")

# ── Global importance ────────────────────────────────────────────────────────────
st.markdown("### 🌍 Global feature importance")
explain("Which features the model relies on most across all transactions. SHAP measures "
        "each feature's average contribution to the prediction; without SHAP we fall back "
        "to the model's built-in importance or permutation importance.")
with st.spinner("Computing importance…"):
    imp = global_importance(model, X_test, feature_cols)
st.caption(f"Method: **{imp['method'].iloc[0]}**")
st.plotly_chart(viz.feature_importance(imp), use_container_width=True)

# ── SHAP summary ─────────────────────────────────────────────────────────────────
if shap_available():
    st.markdown("### 🔬 SHAP summary")
    sv = shap_values_for(model, X_test, feature_cols)
    if sv is not None:
        arr, _ = sv
        st.plotly_chart(viz.shap_summary_bar(arr, feature_cols), use_container_width=True)
    else:
        st.caption("SHAP could not explain this model type; see importance above.")

# ── What-if analysis ─────────────────────────────────────────────────────────────
st.markdown("### 🎛️ What-if analysis")
explain("Move the sliders to change a (standardized) feature value and watch the fraud "
        "probability respond in real time. Values are in standard deviations from the mean "
        "(0 = average, +2 = high, −2 = low).")

base = np.median(np.asarray(X_test), axis=0)
top_feats = imp["feature"].head(6).tolist()
x = base.copy()
cols = st.columns(3)
for i, f in enumerate(top_feats):
    j = feature_cols.index(f)
    x[j] = cols[i % 3].slider(f, -3.0, 3.0, float(base[j]), 0.1)

proba = float(get_proba(model, x.reshape(1, -1))[0])
threshold = prep.get("threshold") or s.get("results", {}).get(s.best_model_name, {}).get("Threshold", 0.5)
factors = top_risk_factors(model, x, feature_cols)

c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(viz.gauge(proba, "Fraud probability"), use_container_width=True)
with c2:
    st.markdown(plain_english_prediction(proba, threshold, factors))
    st.plotly_chart(viz.waterfall_factors(factors), use_container_width=True)
