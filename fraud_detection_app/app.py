"""
Sentinel — AI Fraud Detection Studio
Home / Dashboard.  Run:  streamlit run app.py
"""
import streamlit as st

from utils.config import (setup_page, APP_NAME, APP_TAGLINE, GOLD, glass_card,
                          stat_card)
from utils.data_processor import generate_synthetic_fraud, detect_metadata

setup_page("", icon="🛡️")   # no page header on home; we render a custom hero

s = st.session_state

# ── Hero ────────────────────────────────────────────────────────────────────────
df = s.get("raw_df")
meta = s.get("meta") or {}
n_txn = meta.get("n_rows", 0)
n_fraud = meta.get("fraud_count") or 0
n_models = len(s.get("results", {})) + len(s.get("unsup_results", {}))

st.markdown(
    f"""
    <div class="hero">
      <h1>🛡️ {APP_NAME}</h1>
      <p>{APP_TAGLINE} — detect fraud with or without labels, understand every
         decision, and see the impact in dollars.</p>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(stat_card("Transactions loaded", f"{n_txn:,}", "current dataset"), unsafe_allow_html=True)
c2.markdown(stat_card("Known fraud cases", f"{n_fraud:,}",
                      f"{meta.get('fraud_rate', 0) or 0:.2f}% rate" if meta.get("fraud_rate") is not None else "no labels",
                      tone="red"), unsafe_allow_html=True)
c3.markdown(stat_card("Models trained", f"{n_models}", "this session", tone="green"), unsafe_allow_html=True)
c4.markdown(stat_card("Mode", (s.get("problem_type") or "—").capitalize(),
                      "auto-detected"), unsafe_allow_html=True)

st.markdown("")

# ── Quick actions ─────────────────────────────────────────────────────────────
st.markdown("### 🚀 Quick start")
q1, q2, q3 = st.columns(3)
with q1:
    glass_card("<b>1 · Upload data</b><br><span style='color:#8aa0bd'>Drop a CSV — "
               "we auto-detect the target column, data types, and fraud rate.</span>")
    st.page_link("pages/1_📊_Data_Upload.py", label="Go to Data Upload →")
with q2:
    glass_card("<b>2 · Train models</b><br><span style='color:#8aa0bd'>Compare classic "
               "ML, ensembles, and AutoML — or run anomaly detection if unlabeled.</span>")
    st.page_link("pages/4_📈_Model_Training.py", label="Go to Model Training →")
with q3:
    glass_card("<b>3 · Predict & explain</b><br><span style='color:#8aa0bd'>Score single "
               "or batch transactions and see exactly why each was flagged.</span>")
    st.page_link("pages/5_🎯_Prediction.py", label="Go to Prediction →")

st.divider()

# ── No data? Offer synthetic data ──────────────────────────────────────────────
if df is None:
    st.info("👋 No data loaded yet. Upload a CSV on the **Data Upload** page, or "
            "generate a realistic sample below to explore the app instantly.")
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        n = st.number_input("Sample rows", 1000, 50000, 5000, 1000)
        rate = st.slider("Fraud rate", 0.005, 0.2, 0.03, 0.005)
    with cc2:
        st.markdown("&nbsp;")
        if st.button("✨ Generate synthetic PaySim sample"):
            gen = generate_synthetic_fraud(int(n), float(rate))
            s.raw_df = gen
            s.meta = detect_metadata(gen)
            s.target_col = s.meta["target_col"]
            s.problem_type = s.meta["problem_type"]
            s.amount_col = s.meta["amount_col"]
            st.success(f"Generated {len(gen):,} transactions. Head to **EDA** or **Preprocessing**.")
            st.rerun()
else:
    # System status
    st.markdown("### 📡 System status")
    st.markdown(
        f"- **Dataset:** {n_txn:,} rows × {meta.get('n_cols', '?')} columns"
        f"{' · PaySim schema ✅' if meta.get('is_paysim') else ''}\n"
        f"- **Target:** `{s.get('target_col') or 'none → unsupervised mode'}`\n"
        f"- **Amount column (for $ impact):** `{s.get('amount_col') or '—'}`\n"
        f"- **Preprocessing:** {'done ✅' if s.get('prep') else 'not run yet'}\n"
        f"- **Best model:** {s.get('best_model_name') or '—'}")

st.markdown(
    f"<div style='margin-top:2rem;color:#5b6b82;font-size:.8rem'>"
    f"{APP_NAME} · educational + production-ready · all heavy AutoML/DL deps are "
    f"optional and degrade gracefully.</div>", unsafe_allow_html=True)
