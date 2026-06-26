"""
constants.py — Streamlit-free constants shared across the app.

Kept separate from ``config.py`` (which imports streamlit) so the pure ML and
plotting modules import cleanly and stay unit-testable without a UI runtime.
"""

# ── Branding ──
APP_NAME    = "Sentinel"
APP_TAGLINE = "AI Fraud Detection Studio"
APP_VERSION = "4.4"

# ── Palette ──
NAVY        = "#0a1628"
NAVY_CARD   = "#0f1f38"
NAVY_LIGHT  = "#16304f"
GOLD        = "#f7b731"
GOLD_SOFT   = "#ffd166"
TEXT        = "#e7edf5"
TEXT_MUTED  = "#8aa0bd"
GREEN       = "#2ecc71"
RED         = "#ff5b6e"
AMBER       = "#f7b731"

PLOTLY_COLORS = [GOLD, "#4aa3ff", GREEN, RED, "#a78bfa",
                 "#00d2d3", "#ff9f43", "#54a0ff", "#5f27cd", "#1dd1a1"]

# ── ML config ──
RANDOM_STATE = 42
