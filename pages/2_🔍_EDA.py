"""Page 2 — Exploratory Data Analysis with auto-generated insights."""
import numpy as np
import pandas as pd
import streamlit as st

from utils.config import setup_page, glass_card
from utils.data_processor import _positive_mask
from utils import visualizer as viz

setup_page("Exploratory Data Analysis", "🔍",
           "Interactive charts and plain-English insights about your data.", stage=1)

s = st.session_state
df = s.get("raw_df")
if df is None:
    st.warning("⚠️ Upload data on the **Data Upload** page first.")
    st.stop()

target = s.get("target_col")
pos_mask = _positive_mask(df[target]) if target else None

# ── Auto insights ─────────────────────────────────────────────────────────────
st.markdown("### 💡 Key insights")
insights = []
if target is not None:
    rate = pos_mask.mean() * 100
    insights.append(f"Fraud rate is **{rate:.2f}%** — "
                    + ("highly imbalanced, so we'll weight metrics toward Recall/PR-AUC."
                       if rate < 5 else "reasonably balanced."))
    # Strongest categorical association.
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    best = None
    for c in cat_cols:
        if 1 < df[c].nunique() <= 20:
            g = df.assign(_p=pos_mask.astype(int)).groupby(c)["_p"].mean()
            spread = g.max() - g.min()
            if best is None or spread > best[1]:
                best = (c, spread, g.idxmax(), g.max())
    if best:
        insights.append(f"Fraud concentrates in **{best[0]} = {best[2]}** "
                        f"({best[3]:.1%} fraud there).")
    # Numeric separation.
    num = df.select_dtypes(include=[np.number]).columns
    seps = []
    for c in num:
        if c == target:
            continue
        a, b = df.loc[pos_mask, c], df.loc[~pos_mask, c]
        if a.std() + b.std() > 0:
            seps.append((c, abs(a.mean() - b.mean()) / (a.std() + b.std() + 1e-9)))
    if seps:
        top = sorted(seps, key=lambda x: -x[1])[:3]
        insights.append("Most separating numeric features: "
                        + ", ".join(f"`{c}`" for c, _ in top) + ".")
else:
    insights.append("No label column — use this page to understand the data, then run "
                    "**unsupervised** anomaly detection in Model Training.")
    insights.append(f"Dataset has **{df.shape[0]:,}** rows and "
                    f"**{df.select_dtypes(include=[np.number]).shape[1]}** numeric features to profile.")

for i in insights:
    glass_card("💡 " + i)

# ── Class distribution / fraud by category ─────────────────────────────────────
if target is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.class_distribution(pos_mask.astype(int), target),
                        width="stretch")
    with c2:
        cats = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if 1 < df[c].nunique() <= 20]
        if cats:
            cc = st.selectbox("Fraud rate by category", cats)
            st.plotly_chart(viz.fraud_rate_by_category(df, cc, pos_mask),
                            width="stretch")

# ── Distribution explorer ──────────────────────────────────────────────────────
st.markdown("### 🔬 Distribution explorer")
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
if num_cols:
    col = st.selectbox("Column", num_cols)
    st.plotly_chart(viz.distribution(df[col], col, by=pos_mask.astype(int) if target else None),
                    width="stretch")

# ── Correlation ────────────────────────────────────────────────────────────────
st.markdown("### 🧭 Correlation heatmap")
st.plotly_chart(viz.correlation_heatmap(df), width="stretch")

# ── Time analysis ──────────────────────────────────────────────────────────────
if "step" in df.columns and target is not None:
    st.markdown("### ⏱️ Fraud over time (step)")
    tmp = df.assign(_p=pos_mask.astype(int))
    by_step = tmp.groupby(tmp["step"] // 24)["_p"].mean()
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=by_step.index, y=by_step.values, mode="lines",
                               line=dict(color="#f7b731", width=2)))
    fig.update_layout(xaxis_title="Day (step // 24)", yaxis_title="Fraud rate")
    st.plotly_chart(viz._style(fig, "Fraud rate over time"), width="stretch")
