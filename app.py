"""
Sentinel — AI Fraud Detection Studio
Home.  Each session is fresh (nothing is saved to disk).  Run:  streamlit run app.py
"""
import streamlit as st

from utils.config import (setup_page, APP_NAME, APP_TAGLINE, glass_card, stat_card,
                          STAGES, reset_pipeline_state)

setup_page("", icon="🛡️")   # home is not a pipeline stage
s = st.session_state

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <h1>🛡️ {APP_NAME}</h1>
      <p>{APP_TAGLINE} — a guided pipeline: upload data, explore, preprocess, train,
         predict, and explain. Move through the stages with <b>Previous / Next</b>.</p>
    </div>
    """, unsafe_allow_html=True)

in_progress = s.get("raw_df") is not None
stg = int(s.get("current_stage", 0) or 0)

c1, c2, c3 = st.columns(3)
c1.markdown(stat_card("Stages", f"{len(STAGES)}", "guided pipeline", icon="🧭"), unsafe_allow_html=True)
c2.markdown(stat_card("This session", "In progress" if in_progress else "Empty",
                      STAGES[stg][0] if in_progress else "no data yet",
                      tone="green" if in_progress else "gold", icon="📊"), unsafe_allow_html=True)
c3.markdown(stat_card("Best model", s.get("best_model_name") or "—",
                      "trained" if s.get("best_model_name") else "not trained",
                      tone="green", icon="🤖"), unsafe_allow_html=True)

st.markdown("")

# ── Start / continue ────────────────────────────────────────────────────────────
a1, a2 = st.columns(2)
with a1:
    glass_card("<b>🚀 Start a new analysis</b><br><span style='color:#8aa0bd'>Begin a fresh "
               "pipeline from Data Upload. Note: nothing is saved — closing the app "
               "clears the session.</span>")
    if st.button("Start new analysis ➡", type="primary", use_container_width=True):
        reset_pipeline_state()
        st.switch_page(STAGES[0][1])
with a2:
    glass_card("<b>▶ Continue this session</b><br><span style='color:#8aa0bd'>Jump back to "
               "where you are in the current pipeline.</span>")
    if st.button("Continue ▶", disabled=not in_progress, use_container_width=True):
        st.switch_page(STAGES[stg][1])
    if not in_progress:
        st.caption("Start a new analysis first.")

st.divider()

# ── The pipeline at a glance ────────────────────────────────────────────────────
st.markdown("### 🧭 The pipeline")
cols = st.columns(len(STAGES))
for i, (label, _) in enumerate(STAGES):
    done = i <= int(s.get("max_stage", 0) or 0) and in_progress
    cols[i].markdown(
        f"<div style='text-align:center'><div style='font-size:1.4rem'>"
        f"{'✅' if done else '○'}</div><div style='font-size:.72rem;color:#8aa0bd'>"
        f"{i + 1}. {label}</div></div>", unsafe_allow_html=True)

st.markdown(
    f"<div style='margin-top:2rem;color:#5b6b82;font-size:.8rem'>"
    f"{APP_NAME} · session-only (no data is written to disk) · navigation inside the "
    f"pipeline is guided with Previous / Next.</div>", unsafe_allow_html=True)
