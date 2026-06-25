"""
Sentinel — AI Fraud Detection Studio
Runs home: create, resume, and manage pipeline runs.  Run:  streamlit run app.py
"""
import time
import streamlit as st

from utils.config import (setup_page, APP_NAME, APP_TAGLINE, glass_card, stat_card,
                          STAGES, reset_pipeline_state, apply_loaded_state)
from utils import runs

setup_page("", icon="🛡️")   # home is not a pipeline stage
s = st.session_state


def _start_run(run_id: str, name: str, stage: int = 0):
    s.active_run_id = run_id
    s.active_run_name = name
    runs.save_run(run_id, s)
    st.switch_page(STAGES[stage][1])


# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <h1>🛡️ {APP_NAME}</h1>
      <p>{APP_TAGLINE} — every analysis is a <b>run</b>: a full pipeline you can
         create, pause, resume, and clean up. Pick up exactly where you left off.</p>
    </div>
    """, unsafe_allow_html=True)

all_runs = runs.list_runs()
c1, c2, c3 = st.columns(3)
c1.markdown(stat_card("Saved runs", f"{len(all_runs)}", "on this machine", icon="🗂️"), unsafe_allow_html=True)
c2.markdown(stat_card("With data", f"{sum(1 for r in all_runs if r.get('has_data'))}",
                      "data loaded", tone="green", icon="📊"), unsafe_allow_html=True)
c3.markdown(stat_card("With models", f"{sum(1 for r in all_runs if (r.get('n_models') or 0) > 0)}",
                      "trained", tone="green", icon="🤖"), unsafe_allow_html=True)

# ── Resume the active run ───────────────────────────────────────────────────────
if s.get("active_run_id"):
    st.markdown("### ▶ Current run")
    stg = int(s.get("current_stage", 0) or 0)
    glass_card(f"<b>{s.get('active_run_name') or 'Untitled run'}</b><br>"
               f"<span style='color:#8aa0bd'>You're on stage {stg + 1} · "
               f"{STAGES[stg][0]}.</span>")
    if st.button("▶ Continue current run", type="primary"):
        st.switch_page(STAGES[stg][1])

# ── Create a new run ────────────────────────────────────────────────────────────
st.markdown("### ➕ New run")
nc1, nc2 = st.columns([3, 1])
with nc1:
    new_name = st.text_input("Run name", value=f"Run {time.strftime('%b %d, %H:%M')}",
                             label_visibility="collapsed", placeholder="Name this run…")
with nc2:
    if st.button("Create & start ➡", use_container_width=True):
        reset_pipeline_state()
        rid = runs.new_run(new_name)
        _start_run(rid, new_name.strip() or "Untitled run", stage=0)

st.divider()

# ── Existing runs: open / resume / delete ───────────────────────────────────────
st.markdown("### 🗂️ Your runs")
if not all_runs:
    st.info("No runs yet. Create one above to begin the pipeline.")
else:
    st.caption("Tick runs to delete, or open one to resume it from where you left off.")
    selected = []
    head = st.columns([0.4, 3, 2, 1.6, 1.2])
    for col, label in zip(head, ["", "Run", "Stage", "Updated", ""]):
        col.markdown(f"**{label}**")

    for r in all_runs:
        rid = r["id"]
        col = st.columns([0.4, 3, 2, 1.6, 1.2])
        if col[0].checkbox("select", key=f"del_{rid}", label_visibility="collapsed"):
            selected.append(rid)
        stg = int(r.get("current_stage", 0) or 0)
        mdl = f" · best: {r['best_model_name']}" if r.get("best_model_name") else ""
        col[1].markdown(f"**{r.get('name', 'Untitled')}**  \n"
                        f"<span style='color:#8aa0bd;font-size:.8rem'>"
                        f"{(r.get('problem_type') or 'no data').capitalize()}"
                        f" · {r.get('n_models', 0)} model(s){mdl}</span>",
                        unsafe_allow_html=True)
        col[2].markdown(f"{stg + 1}. {STAGES[stg][0]}")
        col[3].markdown(f"<span style='color:#8aa0bd;font-size:.8rem'>"
                        f"{time.strftime('%b %d, %H:%M', time.localtime(r.get('updated', 0)))}</span>",
                        unsafe_allow_html=True)
        if col[4].button("Open ▶", key=f"open_{rid}", use_container_width=True):
            reset_pipeline_state()
            apply_loaded_state(runs.load_run(rid))
            _start_run(rid, r.get("name", "Untitled run"), stage=stg)

    st.markdown("")
    dc1, dc2 = st.columns([1, 3])
    with dc1:
        if st.button(f"🗑️ Delete selected ({len(selected)})", disabled=not selected,
                     use_container_width=True):
            runs.delete_runs(selected)
            if s.get("active_run_id") in selected:
                s.active_run_id = None
                s.active_run_name = None
            st.rerun()
    with dc2:
        if selected:
            st.caption("⚠️ Deletes the data, models, and everything saved for those runs.")

st.markdown(
    f"<div style='margin-top:2rem;color:#5b6b82;font-size:.8rem'>"
    f"{APP_NAME} · each run is fully self-contained and saved locally · "
    f"navigation inside a run is guided (Previous / Next).</div>",
    unsafe_allow_html=True)
