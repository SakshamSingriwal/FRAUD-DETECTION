"""Page 7 — Executive Dashboard, Fraud Pattern Library & Learning Center."""
import numpy as np
import pandas as pd
import streamlit as st

from utils.config import setup_page, stat_card, glass_card
from utils.model_explainer import METRIC_HELP

setup_page("Dashboard", "📊",
           "Executive summary, fraud-pattern library, and a plain-English learning center.")

s = st.session_state

tab_sum, tab_patterns, tab_learn = st.tabs(
    ["📈 Summary", "🧩 Fraud Pattern Library", "🎓 Learning Center"])

# ── Summary ───────────────────────────────────────────────────────────────────
with tab_sum:
    meta = s.get("meta") or {}
    results = s.get("results", {})
    best = s.get("best_model_name")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Transactions", f"{meta.get('n_rows', 0):,}"), unsafe_allow_html=True)
    c2.markdown(stat_card("Fraud cases", f"{meta.get('fraud_count') or 0:,}",
                          tone="red"), unsafe_allow_html=True)
    c3.markdown(stat_card("Models trained", f"{len(results) + len(s.get('unsup_results', {}))}",
                          tone="green"), unsafe_allow_html=True)
    c4.markdown(stat_card("Best model", best or "—", tone="gold"), unsafe_allow_html=True)

    if best and best in results:
        r = results[best]
        st.markdown("#### 🏆 Best model scorecard")
        m = st.columns(6)
        for col, k in zip(m, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]):
            col.metric(k, r.get(k))
    else:
        st.info("Train a model to populate the scorecard.")

    # Lightweight data-drift check: compare two halves of the dataset.
    df = s.get("raw_df")
    if df is not None:
        st.markdown("#### 📉 Data drift snapshot")
        st.caption("Compares the mean of each numeric feature between the first and second "
                   "half of the dataset (a quick proxy for distribution drift over time).")
        num = df.select_dtypes(include=[np.number])
        if len(num) > 10 and num.shape[1] > 0:
            half = len(num) // 2
            a, b = num.iloc[:half].mean(), num.iloc[half:].mean()
            drift = ((b - a) / (a.abs() + 1e-9) * 100).abs().sort_values(ascending=False)
            drift_df = pd.DataFrame({"feature": drift.index, "mean shift %": drift.round(1).values}).head(10)
            st.dataframe(drift_df, use_container_width=True)
            flagged = drift_df[drift_df["mean shift %"] > 25]
            if len(flagged):
                st.warning(f"⚠️ Possible drift in: {', '.join(flagged['feature'])} "
                           "(>25% mean shift). Consider retraining.")
            else:
                st.success("No major drift detected between dataset halves.")

# ── Fraud Pattern Library ──────────────────────────────────────────────────────
with tab_patterns:
    st.markdown("Common money-laundering / fraud patterns. When PaySim data is loaded, "
                "Sentinel checks your data for each pattern automatically.")
    df = s.get("raw_df")
    target = s.get("target_col")

    PATTERNS = [
        ("💸 Account draining",
         "Origin account emptied in one transfer (new balance = 0, old balance > 0).",
         lambda d: ((d.get("newbalanceOrig", 1) == 0) & (d.get("oldbalanceOrg", 0) > 0)).mean()
         if "newbalanceOrig" in d else None),
        ("🔁 Transfer-then-cashout",
         "Funds moved via TRANSFER then withdrawn via CASH_OUT — classic layering.",
         lambda d: (d["type"].isin(["TRANSFER", "CASH_OUT"]).mean()) if "type" in d else None),
        ("🏦 Zero-balance destination",
         "Destination account starts at zero balance (mule / throwaway account).",
         lambda d: ((d.get("oldbalanceDest", 1) == 0)).mean() if "oldbalanceDest" in d else None),
        ("🧮 Balance mismatch",
         "Reported balances don't reconcile with the amount moved.",
         lambda d: ((d.get("oldbalanceOrg", 0) - d.get("amount", 0) - d.get("newbalanceOrig", 0)).abs() > 1).mean()
         if "amount" in d else None),
    ]
    for title, desc, fn in PATTERNS:
        share = None
        try:
            share = fn(df) if df is not None else None
        except Exception:
            share = None
        extra = f"<br><span style='color:#f7b731'>Present in {share:.1%} of your rows</span>" \
                if share is not None else ""
        glass_card(f"<b>{title}</b><br><span style='color:#8aa0bd'>{desc}</span>{extra}")

# ── Learning Center ────────────────────────────────────────────────────────────
with tab_learn:
    st.markdown("#### 📐 Metric glossary")
    for metric, (what, why) in METRIC_HELP.items():
        with st.expander(metric):
            st.markdown(f"**What it means:** {what}\n\n**Why it matters:** {why}")

    st.markdown("#### 🧠 Concepts in plain English")
    glass_card("<b>Class imbalance</b><br><span style='color:#8aa0bd'>Fraud is rare (often "
               "<1%). A model can score 99% accuracy by calling everything legit — useless. "
               "We fix this with <b>SMOTE</b> (synthesizing fraud examples) or class weights, "
               "and judge models on Recall / PR-AUC, not accuracy.</span>")
    glass_card("<b>Supervised vs unsupervised</b><br><span style='color:#8aa0bd'>Supervised "
               "learns from labeled fraud/not-fraud examples. Unsupervised needs no labels — "
               "it learns 'normal' and flags outliers. Sentinel auto-picks based on your data.</span>")
    glass_card("<b>Decision threshold</b><br><span style='color:#8aa0bd'>Models output a "
               "probability; the threshold turns it into a yes/no. Lower threshold = catch more "
               "fraud but more false alarms. We tune it on a validation split to minimize your "
               "cost model, then report on a held-out test set.</span>")
    glass_card("<b>Explainability (SHAP / LIME)</b><br><span style='color:#8aa0bd'>These tools "
               "attribute a prediction to individual features — turning a black box into 'this "
               "was flagged because the origin account was emptied'. Essential for trust.</span>")
