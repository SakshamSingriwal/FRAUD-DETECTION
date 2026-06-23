"""
visualizer.py — Interactive Plotly charts with the Sentinel dark/gold theme.

Every function returns a ``plotly.graph_objects.Figure`` so pages just call
``st.plotly_chart(fig, use_container_width=True)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.decomposition import PCA

from .constants import GOLD, GREEN, RED, TEXT, TEXT_MUTED, PLOTLY_COLORS, NAVY

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,22,40,0.35)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=46, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(138,160,189,0.12)", zerolinecolor="rgba(138,160,189,0.2)"),
    yaxis=dict(gridcolor="rgba(138,160,189,0.12)", zerolinecolor="rgba(138,160,189,0.2)"),
)


def _style(fig, title=None):
    fig.update_layout(**_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color=TEXT)))
    return fig


def gauge(value, title="Score", vmax=1.0, good_high=True):
    color = GOLD if 0.4 <= value / vmax <= 0.7 else (GREEN if (value / vmax > 0.7) == good_high else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"valueformat": ".3f" if vmax <= 1 else ".0f"},
        gauge={"axis": {"range": [0, vmax]}, "bar": {"color": color},
               "bgcolor": "rgba(255,255,255,0.04)",
               "borderwidth": 0,
               "steps": [{"range": [0, vmax * 0.5], "color": "rgba(255,91,110,0.15)"},
                         {"range": [vmax * 0.5, vmax * 0.8], "color": "rgba(247,183,49,0.15)"},
                         {"range": [vmax * 0.8, vmax], "color": "rgba(46,204,113,0.15)"}]},
        title={"text": title, "font": {"size": 13, "color": TEXT_MUTED}}))
    fig.update_layout(height=220, **{k: v for k, v in _LAYOUT.items() if k in ("paper_bgcolor", "font", "margin")})
    return fig


def class_distribution(y, target="target"):
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["Legit (0)", "Fraud (1)"] if set(counts.index) <= {0, 1} else [str(i) for i in counts.index]
    fig = go.Figure(go.Bar(x=labels, y=counts.values,
                           marker_color=[GREEN, RED][:len(counts)] or GOLD,
                           text=[f"{v:,}" for v in counts.values], textposition="outside"))
    return _style(fig, f"{target} distribution")


def confusion(cm, title="Confusion Matrix"):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="YlOrBr",
                    x=["Pred Legit", "Pred Fraud"], y=["Actual Legit", "Actual Fraud"])
    fig.update_coloraxes(showscale=False)
    return _style(fig, title)


def roc_curves(results, y_test):
    fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        if "y_proba" not in r:
            continue
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({r.get('ROC-AUC', 0):.3f})",
                                 line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=TEXT_MUTED, dash="dash"),
                             showlegend=False))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _style(fig, "ROC Curves")


def pr_curves(results, y_test):
    fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        if "y_proba" not in r:
            continue
        prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
        fig.add_trace(go.Scatter(x=rec, y=prec, name=f"{name} ({r.get('PR-AUC', 0):.3f})",
                                 line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=2.5)))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return _style(fig, "Precision-Recall Curves")


def radar(results, metrics=("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC")):
    fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        if "error" in r:
            continue
        vals = [r.get(m, 0) or 0 for m in metrics]
        vals += vals[:1]
        fig.add_trace(go.Scatterpolar(r=vals, theta=list(metrics) + [metrics[0]],
                                      name=name, fill="toself",
                                      line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)])))
    fig.update_layout(polar=dict(bgcolor="rgba(10,22,40,0.3)",
                                 radialaxis=dict(range=[0, 1], gridcolor="rgba(138,160,189,0.15)"),
                                 angularaxis=dict(gridcolor="rgba(138,160,189,0.15)")))
    return _style(fig, "Multi-metric comparison")


def feature_importance(imp_df, top=12, title="Feature Importance"):
    d = imp_df.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h",
                           marker_color=GOLD))
    return _style(fig, title)


def probability_hist(proba, threshold=0.5, title="Score Distribution"):
    fig = go.Figure(go.Histogram(x=proba, nbinsx=50, marker_color="#4aa3ff"))
    fig.add_vline(x=threshold, line=dict(color=RED, dash="dash", width=2),
                  annotation_text=f"thr {threshold:.2f}")
    fig.update_layout(xaxis_title="Score", yaxis_title="Count")
    return _style(fig, title)


def correlation_heatmap(df, max_cols=25):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > max_cols:
        num = num.iloc[:, :max_cols]
    fig = px.imshow(num.corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    return _style(fig, "Correlation heatmap")


def distribution(series, name, by=None):
    df = pd.DataFrame({name: series})
    if by is not None:
        df["class"] = np.asarray(by)
        fig = px.histogram(df, x=name, color="class", barmode="overlay", nbins=50,
                           color_discrete_sequence=[GREEN, RED])
    else:
        fig = px.histogram(df, x=name, nbins=50, color_discrete_sequence=[GOLD])
    return _style(fig, f"Distribution · {name}")


def fraud_rate_by_category(df, cat_col, target_pos_mask):
    g = df.assign(_pos=np.asarray(target_pos_mask).astype(int)).groupby(cat_col)["_pos"].mean() \
        .sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker_color=RED,
                           text=[f"{v:.1%}" for v in g.values], textposition="outside"))
    fig.update_layout(yaxis_title="Fraud rate")
    return _style(fig, f"Fraud rate by {cat_col}")


def anomaly_scatter(X, flagged, risk):
    """2D PCA projection of normal vs anomalous points."""
    X = np.asarray(X)
    if X.shape[1] > 2:
        X2 = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        X2 = np.column_stack([X, np.zeros(len(X))]) if X.shape[1] == 1 else X
    df = pd.DataFrame({"PC1": X2[:, 0], "PC2": X2[:, 1],
                       "status": np.where(np.asarray(flagged) == 1, "Anomaly", "Normal"),
                       "risk": np.asarray(risk)})
    fig = px.scatter(df, x="PC1", y="PC2", color="status", size="risk", size_max=14,
                     color_discrete_map={"Anomaly": RED, "Normal": "#4aa3ff"}, opacity=0.7)
    return _style(fig, "Normal vs anomalous (PCA projection)")


def shap_summary_bar(shap_arr, feature_cols, top=12):
    imp = np.abs(np.asarray(shap_arr)).mean(axis=0)
    d = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values(
        "importance", ascending=True).tail(top)
    fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h", marker_color=GOLD))
    return _style(fig, "SHAP mean |impact|")


def waterfall_factors(factors, base=0.5):
    """Local explanation as a simple impact bar chart."""
    f = list(reversed(factors))
    colors = [RED if x["impact"] > 0 else GREEN for x in f]
    fig = go.Figure(go.Bar(x=[x["impact"] for x in f], y=[x["feature"] for x in f],
                           orientation="h", marker_color=colors))
    fig.update_layout(xaxis_title="Contribution to fraud score")
    return _style(fig, "Why this prediction")
