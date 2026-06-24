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


# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULT_STATE = {
    "raw_df":          None,   # uploaded DataFrame
    "meta":            None,   # detected metadata dict
    "target_col":      None,   # chosen target (None => unsupervised)
    "problem_type":    None,   # "supervised" | "unsupervised"
    "prep":            None,   # preprocessing result dict
    "results":         {},     # supervised model results
    "unsup_results":   {},     # unsupervised model results
    "best_model_name": None,
    "best_model":      None,
    "scaler":          None,
    "feature_cols":    None,
}


def init_state() -> None:
    for k, v in _DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Theme / CSS ────────────────────────────────────────────────────────────────
def _css() -> str:
    css_path = os.path.join(ASSETS_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


def setup_page(title: str, icon: str = "🛡️", subtitle: str = "") -> None:
    """Call once at the top of every page."""
    st.set_page_config(page_title=f"{APP_NAME} · {title}", page_icon=icon,
                        layout="wide", initial_sidebar_state="expanded")
    init_state()
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)
    _sidebar_brand()
    if title:
        page_header(title, subtitle, icon)


def _sidebar_brand() -> None:
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
        _pipeline_status()


def _pipeline_status() -> None:
    """Compact 'where am I in the workflow' indicator."""
    s = st.session_state
    steps = [
        ("Data",     s.get("raw_df") is not None),
        ("Prep",     s.get("prep") is not None),
        ("Models",   bool(s.get("results")) or bool(s.get("unsup_results"))),
        ("Predict",  s.get("best_model") is not None or bool(s.get("unsup_results"))),
    ]
    chips = "".join(
        f'<span class="step {"step-on" if done else "step-off"}">{name}</span>'
        for name, done in steps
    )
    st.markdown(f'<div class="steps">{chips}</div>', unsafe_allow_html=True)
    if s.get("problem_type"):
        mode = s["problem_type"].capitalize()
        st.markdown(f'<div class="mode-badge">{mode} mode</div>', unsafe_allow_html=True)


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
