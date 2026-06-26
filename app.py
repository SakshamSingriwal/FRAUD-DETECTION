"""
Sentinel — AI Fraud Detection Studio
Home.  Each session is fresh (nothing is saved to disk).  Run:  streamlit run app.py
"""
import streamlit as st

from utils.config import (setup_page, APP_NAME, APP_TAGLINE, glass_card,
                          STAGES, reset_pipeline_state)

setup_page("", icon="🛡️")   # home is not a pipeline stage
s = st.session_state

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <h1>🛡️ {APP_NAME}</h1>
      <p>{APP_TAGLINE} — a guided, {len(STAGES)}-step pipeline: upload data, explore,
         preprocess, train, predict, and explain. Move through the stages with
         <b>Previous / Next</b>.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Start ───────────────────────────────────────────────────────────────────────
glass_card("<b>🚀 Start a new analysis</b><br><span style='color:#8aa0bd'>Begin a fresh "
           "pipeline from Data Upload. This is a session-only studio — nothing is written "
           "to disk, so closing the app clears everything.</span>")
if st.button("Start new analysis ➡", type="primary", use_container_width=True):
    reset_pipeline_state()
    st.switch_page(STAGES[0][1])

st.divider()

# ── The pipeline at a glance ────────────────────────────────────────────────────
st.markdown("### 🧭 The pipeline")
cols = st.columns(len(STAGES))
for i, (label, _) in enumerate(STAGES):
    cols[i].markdown(
        f"<div style='text-align:center'><div style='font-size:1.5rem;color:#f7b731'>"
        f"{i + 1}</div><div style='font-size:.72rem;color:#8aa0bd'>{label}</div></div>",
        unsafe_allow_html=True)

st.markdown(
    f"<div style='margin-top:2rem;color:#5b6b82;font-size:.8rem'>"
    f"{APP_NAME} · session-only (no data is written to disk) · navigation inside the "
    f"pipeline is guided with Previous / Next.</div>", unsafe_allow_html=True)
