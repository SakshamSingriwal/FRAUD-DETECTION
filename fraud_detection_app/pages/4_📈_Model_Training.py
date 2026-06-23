"""Page 4 — Model Training & Comparison (supervised) + Anomaly Detection (unsupervised)."""
import numpy as np
import pandas as pd
import streamlit as st

from utils.config import setup_page, stat_card, explain
from utils.model_trainer import (list_supervised_models, train_models, train_automl,
                                 AUTOML_NAMES, save_artifacts, ModelNotPersistableError,
                                 business_impact)
from utils.data_processor import _positive_mask
from utils import unsupervised as un
from utils import visualizer as viz

setup_page("Model Training", "📈",
           "Train, compare, and crown the best model — or detect anomalies with no labels.")

s = st.session_state
prep = s.get("prep")
if not prep:
    st.warning("⚠️ Run **Preprocessing** first.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# UNSUPERVISED MODE
# ════════════════════════════════════════════════════════════════════════════════
if prep.get("unsupervised"):
    st.markdown("### 🟣 Unsupervised anomaly detection")
    explain("No labels were provided, so we learn what 'normal' looks like and flag "
            "outliers. Each transaction gets a **0–100 risk score**; the threshold is "
            "set automatically at the expected anomaly rate (contamination).")
    dets = st.multiselect("Detectors", un.available_detectors(),
                          default=["Isolation Forest"])
    contamination = st.slider("Expected anomaly rate", 0.01, 0.2, 0.05, 0.01)

    if st.button("🚀 Run anomaly detection") and dets:
        s.unsup_results = {}
        prog = st.progress(0.0)
        for i, name in enumerate(dets):
            try:
                s.unsup_results[name] = un.run_detector(name, prep["X"], contamination)
            except Exception as e:
                st.warning(f"{name} failed: {e}")
            prog.progress((i + 1) / len(dets))
        st.success("✅ Done.")

    if s.get("unsup_results"):
        names = list(s.unsup_results.keys())
        pick = st.selectbox("View detector", names)
        r = s.unsup_results[pick]
        a, b, c = st.columns(3)
        a.markdown(stat_card("Flagged", f"{r['n_flagged']:,}", f"{pick}", tone="red"), unsafe_allow_html=True)
        b.markdown(stat_card("Flag rate", f"{r['n_flagged']/len(r['flagged'])*100:.2f}%"), unsafe_allow_html=True)
        c.markdown(stat_card("Threshold", f"{r['threshold']:.3f}", "auto", tone="green"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.anomaly_scatter(prep["X"], r["flagged"], r["risk"]),
                            use_container_width=True)
        with c2:
            st.plotly_chart(viz.probability_hist(r["scores"], r["threshold"], "Anomaly score"),
                            use_container_width=True)
        # If labels happen to exist, show how well anomalies align with fraud.
        if s.get("raw_df") is not None and s.get("meta", {}).get("target_col"):
            from sklearn.metrics import roc_auc_score
            y = _positive_mask(s.raw_df[s.meta["target_col"]]).astype(int).to_numpy()
            try:
                st.info(f"📐 Validation (labels found): anomaly-score ROC-AUC vs true fraud = "
                        f"**{roc_auc_score(y, r['scores']):.3f}**")
            except Exception:
                pass
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# SUPERVISED MODE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("### 🤖 Select models")
classic = list_supervised_models()
all_models = classic + list(AUTOML_NAMES)

select_all = st.checkbox("Select all classic models")
cols = st.columns(3)
selected = []
for i, name in enumerate(all_models):
    is_automl = name in AUTOML_NAMES
    default = select_all and not is_automl
    if cols[i % 3].checkbox(name + (" ⚡" if is_automl else ""), value=default, key=f"m_{name}"):
        selected.append(name)

c1, c2 = st.columns([1, 2])
with c1:
    time_limit = st.slider("AutoML time budget (s)", 30, 300, 90, 10)

# Cost model for threshold tuning + business impact.
with st.expander("💰 Cost model (drives threshold tuning & ROI)"):
    cc1, cc2, cc3 = st.columns(3)
    s.cost_fn = cc1.number_input("Cost of a missed fraud (FN)", 0.0, 1e6, float(s.get("cost_fn", 1.0)))
    s.cost_fp = cc2.number_input("Cost of a false alarm (FP)", 0.0, 1e6, float(s.get("cost_fp", 0.1)))
    fp_review = cc3.number_input("$ to review one false alarm", 0.0, 1e5, 5.0)

if not selected:
    st.info("Select at least one model.")
    st.stop()

if st.button("🚀 Train selected models"):
    prog = st.progress(0.0)
    status = st.empty()
    classic_sel = [m for m in selected if m not in AUTOML_NAMES]
    automl_sel = [m for m in selected if m in AUTOML_NAMES]
    results = {}

    if classic_sel:
        def cb(frac, name):
            prog.progress(frac * len(classic_sel) / len(selected))
            status.text(f"✅ Trained: {name}")
        results.update(train_models(classic_sel, prep, s.cost_fn, s.cost_fp, cb))

    for j, name in enumerate(automl_sel):
        status.text(f"⏳ Running {name} (≤{time_limit}s)…")
        r = train_automl(name, prep, time_limit, s.cost_fn, s.cost_fp)
        if r:
            results[name] = r
        else:
            st.warning(f"{name} unavailable or failed — skipped.")
        prog.progress((len(classic_sel) + j + 1) / len(selected))

    prog.progress(1.0)
    status.text("✅ Training complete.")
    s.results = results

    best = max((n for n, r in results.items() if "ROC-AUC" in r and not np.isnan(r["ROC-AUC"])),
               key=lambda n: results[n]["ROC-AUC"], default=None)
    if best:
        s.best_model_name = best
        s.best_model = results[best]["model"]
        try:
            save_artifacts(s.best_model, prep["scaler"], prep["feature_cols"], best,
                           extra={"feature_mode": prep["feature_mode"],
                                  "is_paysim": prep["is_paysim"],
                                  "threshold": results[best]["Threshold"],
                                  "target_col": prep["target_col"]})
            st.success(f"🏆 Best model: **{best}** — saved to `models/`.")
        except ModelNotPersistableError as e:
            st.warning(f"🏆 Best model: **{best}**. {e}")

results = s.get("results", {})
if not results:
    st.info("Train models to see the comparison.")
    st.stop()

best = s.get("best_model_name")
yte = prep["y_test"]

# ── Comparison table ─────────────────────────────────────────────────────────────
st.markdown("### 🏁 Comparison")
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "LogLoss", "Threshold", "train_time"]
rows = []
for name, r in results.items():
    if "error" in r:
        rows.append({"Model": name, "Error": r["error"]})
    else:
        rows.append({"Model": name + (" 🏆" if name == best else ""),
                     **{m: r.get(m) for m in metric_cols}})
table = pd.DataFrame(rows)
if "ROC-AUC" in table:
    table = table.sort_values("ROC-AUC", ascending=False)
st.dataframe(table, use_container_width=True)

valid = {n: r for n, r in results.items() if "y_proba" in r}
if valid:
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.roc_curves(valid, yte), use_container_width=True)
    c2.plotly_chart(viz.pr_curves(valid, yte), use_container_width=True)
    st.plotly_chart(viz.radar(valid), use_container_width=True)

# ── Business impact for the best model ───────────────────────────────────────────
if best and best in results:
    st.markdown("### 💰 Business impact — best model")
    explain("We translate the test-set confusion matrix into money: fraud **caught** "
            "(true positives), fraud **missed** (false negatives), and the cost of "
            "reviewing **false alarms**. Net savings = caught − review cost.")
    amounts = None
    if s.get("amount_col") and s.get("raw_df") is not None:
        # Align amounts to the test rows is non-trivial post-split; use the average
        # fraud amount as a transparent proxy unless per-row amounts are wired.
        avg_amt = float(pd.to_numeric(s.raw_df[s.amount_col], errors="coerce").dropna().mean())
    else:
        avg_amt = 1000.0
    bi = business_impact(yte, results[best]["y_pred"], amounts=amounts,
                         fp_review_cost=fp_review, avg_fraud_amount=avg_amt)
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(stat_card("Fraud caught", f"${bi['fraud_caught']:,.0f}",
                          f"{bi['n_tp']} txns", tone="green"), unsafe_allow_html=True)
    b2.markdown(stat_card("Fraud missed", f"${bi['fraud_missed']:,.0f}",
                          f"{bi['n_fn']} txns", tone="red"), unsafe_allow_html=True)
    b3.markdown(stat_card("False-alarm cost", f"${bi['fp_cost']:,.0f}",
                          f"{bi['fp_count']} alarms"), unsafe_allow_html=True)
    b4.markdown(stat_card("Net savings", f"${bi['net_savings']:,.0f}",
                          f"ROI {bi['roi_pct']}%", tone="green"), unsafe_allow_html=True)
    st.caption("ℹ️ Dollar figures use the dataset's average amount as a proxy. Wire per-row "
               "amounts for exact values.")

# ── Per-model details ────────────────────────────────────────────────────────────
st.markdown("### 🔎 Per-model details")
for name, r in results.items():
    if "error" in r:
        st.error(f"{name}: {r['error']}")
        continue
    with st.expander(("🏆 " if name == best else "") + name):
        m = st.columns(5)
        for col, key in zip(m, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
            col.metric(key, r.get(key))
        st.plotly_chart(viz.confusion(r["confusion_matrix"], f"{name} · confusion matrix"),
                        use_container_width=True)
