"""
config.py — Central configuration, theme, and shared Streamlit helpers.

Everything visual (palette, glass-morphism CSS) and every shared constant lives
here so the rest of the app stays DRY. Import ``setup_page`` at the top of every
page to get a consistent look, sidebar brand, and session-state bootstrap.
"""
from __future__ import annotations

import os
import streamlit as st
import streamlit.components.v1 as components

# Re-export the Streamlit-free constants so existing `from .config import GOLD`
# style imports keep working.
from .constants import (  # noqa: F401
    APP_NAME, APP_TAGLINE, APP_VERSION,
    NAVY, NAVY_CARD, NAVY_LIGHT, GOLD, GOLD_SOFT, TEXT, TEXT_MUTED, GREEN, RED, AMBER,
    PLOTLY_COLORS, RANDOM_STATE,
)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


# ── Pipeline stages (the guided wizard) ─────────────────────────────────────────
# (label, page file). Order defines Previous / Next.
STAGES = [
    ("Data Upload",        "pages/1_📊_Data_Upload.py"),
    ("EDA",                "pages/2_🔍_EDA.py"),
    ("Preprocessing",      "pages/3_⚙️_Preprocessing.py"),
    ("Model Training",     "pages/4_📈_Model_Training.py"),
    ("Prediction",         "pages/5_🎯_Prediction.py"),
    ("Explainability",     "pages/6_📚_Model_Explainability.py"),
    ("Dashboard",          "pages/7_📊_Dashboard.py"),
]
HOME_PAGE = "app.py"


# ── Session state ──────────────────────────────────────────────────────────────
_PIPELINE_KEYS = {
    "raw_df":          None,   # uploaded DataFrame
    "meta":            None,   # detected metadata dict
    "target_col":      None,   # chosen target (None => unsupervised)
    "problem_type":    None,   # "supervised" | "unsupervised"
    "prep":            None,   # preprocessing result dict
    "results":         {},     # supervised model results
    "unsup_results":   {},     # unsupervised model results
    "best_model_name": None,
    "best_model":      None,
    "selected_model_name": None,   # model chosen for prediction / deployment
    "selected_model":  None,
    "scaler":          None,
    "feature_cols":    None,
    "current_stage":   0,      # wizard position
    "max_stage":       0,      # furthest stage reached (for the stepper)
}
_DEFAULT_STATE = {**_PIPELINE_KEYS, "active_run_id": None, "active_run_name": None}


def init_state() -> None:
    for k, v in _DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_pipeline_state() -> None:
    """Clear the pipeline (used when starting a brand-new run)."""
    for k, v in _PIPELINE_KEYS.items():
        st.session_state[k] = ({} if isinstance(v, dict) else v)


def apply_loaded_state(state: dict) -> None:
    """Populate session_state from a saved run, then rebuild model objects."""
    for k, v in state.items():
        st.session_state[k] = v
    s = st.session_state
    res = s.get("results") or {}
    bn = s.get("best_model_name")
    if bn in res and "model" in res[bn]:
        s["best_model"] = res[bn]["model"]
    sn = s.get("selected_model_name") or bn
    if sn in res and "model" in res[sn]:
        s["selected_model_name"], s["selected_model"] = sn, res[sn]["model"]


def active_model():
    """The model chosen for prediction/deployment (falls back to best)."""
    s = st.session_state
    return s.get("selected_model") or s.get("best_model")


def autosave() -> None:
    """Persist the current run if one is active."""
    from utils import runs
    rid = st.session_state.get("active_run_id")
    if rid:
        runs.save_run(rid, st.session_state)


# ── Theme / CSS ────────────────────────────────────────────────────────────────
def _css() -> str:
    css_path = os.path.join(ASSETS_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


def setup_page(title: str, icon: str = "🛡️", subtitle: str = "",
               stage: int | None = None) -> None:
    """Call once at the top of every page.

    ``stage`` (0-based) marks a pipeline page. When set, the page requires an
    active run, shows the locked wizard stepper, and is reachable only via the
    Previous/Next buttons (the native page nav is hidden in CSS).
    """
    st.set_page_config(page_title=f"{APP_NAME} · {title}", page_icon=icon,
                        layout="wide", initial_sidebar_state="expanded")
    init_state()
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)
    _sidebar_brand(stage)
    if stage is not None:
        _require_run()
        st.session_state["current_stage"] = stage
    if title:
        page_header(title, subtitle, icon)


def _sidebar_brand(stage: int | None = None) -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand">
              <div class="brand-logo">🛡️</div>
              <div>
                <div class="brand-name">{APP_NAME}</div>
                <div class="brand-tag">{APP_TAGLINE} · v{APP_VERSION}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        run_name = st.session_state.get("active_run_name")
        if run_name:
            st.markdown(f'<div class="run-chip">▶ Run: <b>{run_name}</b></div>',
                        unsafe_allow_html=True)
        if stage is not None:
            _wizard_stepper(stage)
            _wizard_nav(stage)
        if st.button("🏠 Runs home", key="wz_home", use_container_width=True):
            autosave()
            st.switch_page(HOME_PAGE)


def _wizard_stepper(current: int) -> None:
    """Locked stepper: shows progress; stages are display-only (no clicking)."""
    s = st.session_state
    max_stage = int(s.get("max_stage", 0) or 0)
    rows = ""
    for i, (label, _) in enumerate(STAGES):
        if i == current:
            cls, mark = "wz-cur", "▶"
        elif i <= max_stage:
            cls, mark = "wz-done", "✓"
        else:
            cls, mark = "wz-todo", "○"
        rows += f'<div class="wz {cls}">{mark} {i + 1}. {label}</div>'
    st.markdown(f'<div class="wz-list">{rows}</div>', unsafe_allow_html=True)
    if s.get("problem_type"):
        st.markdown(f'<div class="mode-badge">{s["problem_type"].capitalize()} mode</div>',
                    unsafe_allow_html=True)


def _require_run() -> None:
    if not st.session_state.get("active_run_id"):
        st.warning("No active run. Open or create one from **Runs home**.")
        if st.button("🏠 Go to Runs home"):
            st.switch_page(HOME_PAGE)
        st.stop()


_STAGE_HINT = {0: "Upload data to continue", 2: "Run preprocessing to continue",
               3: "Train at least one model to continue"}


def _stage_complete(stage: int) -> bool:
    """Whether the user may advance from this stage (gates the Next button)."""
    s = st.session_state
    if stage == 0:
        return s.get("raw_df") is not None
    if stage == 2:
        return s.get("prep") is not None
    if stage == 3:
        return bool(s.get("results")) or bool(s.get("unsup_results"))
    return True


def _wizard_nav(stage: int) -> None:
    """Previous / Next buttons — the only way to move between stages."""
    can_next = _stage_complete(stage)
    n1, n2 = st.columns(2)
    with n1:
        if stage > 0 and st.button("⬅ Prev", key="wz_prev", use_container_width=True):
            _goto(stage - 1)
    with n2:
        if stage < len(STAGES) - 1 and st.button("Next ➡", key="wz_next", type="primary",
                                                 disabled=not can_next, use_container_width=True):
            _goto(stage + 1)
    if stage < len(STAGES) - 1 and not can_next:
        st.caption(f"⛔ {_STAGE_HINT.get(stage, 'Finish this step to continue')}")


def _goto(stage: int) -> None:
    stage = max(0, min(stage, len(STAGES) - 1))
    st.session_state["current_stage"] = stage
    st.session_state["max_stage"] = max(int(st.session_state.get("max_stage", 0) or 0), stage)
    autosave()
    st.switch_page(STAGES[stage][1])


# ── Reusable UI atoms ───────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    st.markdown(
        f'<div class="page-title">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>',
                    unsafe_allow_html=True)


def glass_card(html: str) -> None:
    st.markdown(f'<div class="glass">{html}</div>', unsafe_allow_html=True)


def stat_card(label: str, value: str, sub: str = "", tone: str = "gold",
              icon: str = "") -> str:
    icon_html = f'<div class="stat-icon">{icon}</div>' if icon else ""
    return (
        f'<div class="stat stat-{tone}">{icon_html}'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-sub">{sub}</div></div>'
    )


def explain(text: str, label: str = "💡 Explain") -> None:
    """Collapsible plain-English explainer used throughout the app."""
    with st.expander(label):
        st.markdown(text)


# ── Auto-scroll to results after a button action ────────────────────────────────
# Streamlit appends new output below the fold, so after clicking a button the
# results can land off-screen. `anchor()` drops an invisible target near the
# results; a button handler calls `request_scroll(name)`; `apply_scroll()` (called
# once at the end of the page) smooth-scrolls the parent page to that target.
def anchor(name: str) -> None:
    st.markdown(f'<div id="{name}" style="scroll-margin-top:70px"></div>',
                unsafe_allow_html=True)


def request_scroll(name: str) -> None:
    st.session_state["_scroll_target"] = name


def apply_scroll(name: str | None = None) -> None:
    """Scroll to ``name`` if given (inline use), else to a pending request_scroll
    target (consumed once). Call once near the end of a page, or inline right after
    rendering results."""
    name = name or st.session_state.pop("_scroll_target", None)
    if not name:
        return
    components.html(
        f"""
        <script>
          let tries = 0;
          const timer = setInterval(function() {{
            const el = window.parent.document.getElementById('{name}');
            if (el) {{ el.scrollIntoView({{behavior: 'smooth', block: 'start'}}); clearInterval(timer); }}
            if (++tries > 25) clearInterval(timer);
          }}, 100);
        </script>
        """, height=0)
