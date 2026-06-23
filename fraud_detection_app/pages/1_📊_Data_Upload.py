"""Page 1 — Data Upload & intelligent detection."""
import pandas as pd
import streamlit as st

from utils.config import setup_page, stat_card, explain
from utils.data_processor import (detect_metadata, data_quality_report,
                                  generate_synthetic_fraud)

setup_page("Data Upload", "📊",
           "Drop a CSV — Sentinel auto-detects your target, types, and fraud rate.")

s = st.session_state
_STATUS = {"good": "🟢", "warn": "🟡", "bad": "🔴"}

up = st.file_uploader("Upload training CSV", type="csv",
                      help="Any tabular CSV. A binary fraud/label column is detected automatically.")

col_u, col_g = st.columns([3, 1])
with col_g:
    if st.button("✨ Use synthetic sample"):
        gen = generate_synthetic_fraud(5000, 0.03)
        s.raw_df = gen
        up = None

if up is not None:
    try:
        s.raw_df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
    # reset downstream state
    for k in ("prep", "results", "unsup_results", "best_model", "best_model_name"):
        s[k] = {} if k.endswith("results") else None

df = s.get("raw_df")
if df is None:
    st.info("⬆️ Upload a CSV or generate a sample to begin.")
    st.stop()

# ── Detect metadata ─────────────────────────────────────────────────────────────
meta = detect_metadata(df)
s.meta = meta

# Let the user confirm / override the auto-detected target.
st.markdown("### 🎯 Dataset configuration")
options = ["(none → unsupervised)"] + list(df.columns)
default = df.columns.get_loc(meta["target_col"]) + 1 if meta["target_col"] else 0
choice = st.selectbox("Target column", options, index=default,
                      help="The fraud/label column. Choose '(none)' to run label-free anomaly detection.")
s.target_col = None if choice.startswith("(none") else choice
s.problem_type = "unsupervised" if s.target_col is None else "supervised"

badge = "🟢 Supervised — labels detected" if s.problem_type == "supervised" \
        else "🟣 Unsupervised — no labels, anomaly detection mode"
st.markdown(f"**Detected mode:** {badge}")
explain(
    "**Supervised** learning needs a column telling us which past transactions were "
    "fraud (the *labels*). When that's missing, Sentinel switches to **unsupervised** "
    "anomaly detection — it learns what 'normal' looks like and flags transactions that "
    "stand out, no labels required.")

st.divider()

# ── Quick stats ─────────────────────────────────────────────────────────────────
st.markdown("### 📈 Overview")
g1, g2, g3, g4 = st.columns(4)
g1.markdown(stat_card("Rows", f"{meta['n_rows']:,}"), unsafe_allow_html=True)
g2.markdown(stat_card("Columns", f"{meta['n_cols']}"), unsafe_allow_html=True)
g3.markdown(stat_card("Missing cells", f"{meta['missing_total']:,}",
                      tone=("green" if meta["missing_total"] == 0 else "gold")), unsafe_allow_html=True)
g4.markdown(stat_card("Duplicate rows", f"{meta['duplicates']:,}",
                      tone=("green" if meta["duplicates"] == 0 else "gold")), unsafe_allow_html=True)

if s.problem_type == "supervised":
    f1, f2 = st.columns(2)
    f1.markdown(stat_card("Fraud cases", f"{meta['fraud_count']:,}", tone="red"), unsafe_allow_html=True)
    f2.markdown(stat_card("Fraud rate", f"{meta['fraud_rate']:.2f}%",
                          "highly imbalanced" if (meta['fraud_rate'] or 0) < 5 else "", tone="red"),
                unsafe_allow_html=True)

# ── Data quality report ───────────────────────────────────────────────────────
st.markdown("### 🩺 Data quality report")
for r in data_quality_report(df):
    st.markdown(f"{_STATUS[r['status']]} **{r['check']}** — {r['detail']}")

# ── Preview ───────────────────────────────────────────────────────────────────
with st.expander("📄 Data preview", expanded=True):
    st.dataframe(df.head(200), width="stretch")
with st.expander("📊 Summary statistics"):
    st.dataframe(df.describe(include="all").T, width="stretch")

st.success("✅ Configuration saved. Continue to **EDA** or **Preprocessing**.")
