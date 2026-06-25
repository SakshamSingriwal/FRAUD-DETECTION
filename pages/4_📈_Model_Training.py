"""Page 4 — Model Training & Comparison (supervised) + Anomaly Detection (unsupervised)."""
import numpy as np
import pandas as pd
import streamlit as st

from utils.config import (setup_page, stat_card, explain, anchor, request_scroll,
                          apply_scroll, autosave)
from utils.model_trainer import (list_supervised_models, train_models, train_automl,
                                 AUTOML_NAMES, save_artifacts, ModelNotPersistableError,
                                 detection_summary, pick_best_model)
from utils.model_explainer import METRIC_HELP
from utils.data_processor import _positive_mask
from utils import unsupervised as un
from utils import visualizer as viz

setup_page("Model Training", "📈",
           "Train, compare, and crown the best model — or detect anomalies with no labels.",
           stage=3)

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
        request_scroll("unsup-results")
        autosave()

    anchor("unsup-results")
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
                            width="stretch")
        with c2:
            st.plotly_chart(viz.probability_hist(r["scores"], r["threshold"], "Anomaly score"),
                            width="stretch")
        # If labels happen to exist, show how well anomalies align with fraud.
        if s.get("raw_df") is not None and s.get("meta", {}).get("target_col"):
            from sklearn.metrics import roc_auc_score
            y = _positive_mask(s.raw_df[s.meta["target_col"]]).astype(int).to_numpy()
            try:
                st.info(f"📐 Validation (labels found): anomaly-score ROC-AUC vs true fraud = "
                        f"**{roc_auc_score(y, r['scores']):.3f}**")
            except Exception:
                pass
    apply_scroll()
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# SUPERVISED MODE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("### 🤖 Select models")
classic = list_supervised_models()
all_models = classic + list(AUTOML_NAMES)

# "Select all" toggles each classic model via a callback. (We must write the
# individual checkboxes' session_state here — a keyed checkbox ignores `value=`
# on reruns, so simply recomputing a default would not update them.)
def _toggle_all_classic():
    for nm in classic:
        st.session_state[f"m_{nm}"] = st.session_state["select_all_classic"]

st.checkbox("Select all classic models", key="select_all_classic",
            on_change=_toggle_all_classic)
cols = st.columns(3)
selected = []
for i, name in enumerate(all_models):
    is_automl = name in AUTOML_NAMES
    if cols[i % 3].checkbox(name + (" ⚡" if is_automl else ""), key=f"m_{name}"):
        selected.append(name)

c1, c2 = st.columns([1, 2])
with c1:
    time_limit = st.slider("AutoML time budget (s)", 30, 300, 90, 10)

explain(
    "**How the decision threshold is chosen:** for each model we pick the probability "
    "cut that **maximises F1 on the validation split**, then report all metrics on the "
    "untouched test split. F1 balances catching fraud (recall) against false alarms "
    "(precision) — no business cost assumptions needed.\n\n"
    "**Avoiding under/over-fitting:** models use regularised settings (shallow trees, "
    "leaf-size floors, subsampling, L2); boosters use early stopping on the validation "
    "set. Each model is then labelled **underfit / good / overfit** from its train→test "
    "AUC gap, shown in the comparison below.")

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
        results.update(train_models(classic_sel, prep, progress_cb=cb))

    for j, name in enumerate(automl_sel):
        status.text(f"⏳ Running {name} (≤{time_limit}s)…")
        r = train_automl(name, prep, time_limit)
        if r and "error" not in r:
            results[name] = r
        else:
            reason = r.get("error", "unknown reason") if r else "unknown reason"
            st.warning(f"⚠️ **{name}** skipped — {reason}")
        prog.progress((len(classic_sel) + j + 1) / len(selected))

    prog.progress(1.0)
    status.text("✅ Training complete.")
    s.results = results
    request_scroll("train-results")

    best = pick_best_model(results)   # consistent strength across metrics, not one
    if best:
        s.best_model_name = best
        s.best_model = results[best]["model"]
        s.selected_model_name = best                 # default active model = best
        s.selected_model = results[best]["model"]
        try:
            save_artifacts(s.best_model, prep["scaler"], prep["feature_cols"], best,
                           extra={"feature_mode": prep["feature_mode"],
                                  "is_paysim": prep["is_paysim"],
                                  "threshold": results[best]["Threshold"],
                                  "target_col": prep["target_col"]})
            st.success(f"🏆 Best model: **{best}** — saved to `models/`.")
        except ModelNotPersistableError as e:
            st.warning(f"🏆 Best model: **{best}**. {e}")
    autosave()

results = s.get("results", {})
if not results:
    st.info("Train models to see the comparison.")
    st.stop()

best = s.get("best_model_name")
yte = prep["y_test"]

# ── Comparison table ─────────────────────────────────────────────────────────────
anchor("train-results")
st.markdown("### 🏁 Comparison")
metric_cols = ["Train AUC", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1",
               "LogLoss", "Fit", "Threshold", "train_time"]
rows = []
for name, r in results.items():
    if "error" in r:
        rows.append({"Model": name, "Error": r["error"]})
    else:
        rows.append({"Model": name + (" 🏆" if name == best else ""),
                     **{m: r.get(m) for m in metric_cols}})
table = pd.DataFrame(rows)
if "PR-AUC" in table:
    table = table.sort_values("PR-AUC", ascending=False)
st.dataframe(table, width="stretch")

# ── Active model selector (used by Prediction & Deployment) ──────────────────────
trained = [n for n, r in results.items() if "error" not in r]
if trained:
    cur = s.get("selected_model_name") or best
    idx = trained.index(cur) if cur in trained else 0
    pick = st.selectbox("🎯 Active model for Prediction & Deployment", trained, index=idx,
                        help="Pick which trained model the next stages use. Defaults to the "
                             "best (🏆); switch to compare any other model downstream.")
    s.selected_model = results[pick]["model"]
    if s.get("selected_model_name") != pick:
        s.selected_model_name = pick
        autosave()
    if pick != best:
        st.caption(f"Using **{pick}** downstream (best is **{best}**).")
st.caption("🏆 **Best model** = best *average rank* across PR-AUC, F1, ROC-AUC and LogLoss — "
           "rewards consistent strength, so no model wins on a single noise-level metric. "
           "**Train AUC vs ROC-AUC** is the generalisation check (large gap ⇒ overfitting; "
           "both low ⇒ underfitting); the **Fit** column states the verdict per model.")

with st.expander("📖 What does each metric mean? (plain English — so you can explain it)"):
    st.markdown("Every column above, defined simply. **TP/FP/FN/TN** = true/false "
                "positives/negatives (a positive = a transaction flagged as fraud).")
    for m in ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "LogLoss",
              "Train AUC", "Fit", "Threshold"]:
        what, why, good = METRIC_HELP[m]
        st.markdown(f"- **{m}** — {what} _{why}_ · **Good:** {good}")

valid = {n: r for n, r in results.items() if "y_proba" in r}
if valid:
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.roc_curves(valid, yte), width="stretch")
    c2.plotly_chart(viz.pr_curves(valid, yte), width="stretch")
    st.plotly_chart(viz.radar(valid), width="stretch")

# ── Detection summary for the best model ─────────────────────────────────────────
if best and best in results:
    st.markdown("### 🎯 Detection summary — best model")
    explain("Directly observed outcomes on the held-out **test set** (no cost assumptions): "
            "how much real fraud we caught vs missed, and how many legit transactions were "
            "wrongly flagged. **Detection rate** = recall; **false-alarm rate** = share of "
            "legit transactions flagged.")
    ds = detection_summary(yte, results[best]["y_pred"])
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(stat_card("Fraud caught", f"{ds['tp']:,}",
                          f"of {ds['tp'] + ds['fn']:,} real frauds", tone="green"), unsafe_allow_html=True)
    b2.markdown(stat_card("Fraud missed", f"{ds['fn']:,}", "false negatives", tone="red"),
                unsafe_allow_html=True)
    b3.markdown(stat_card("Detection rate", f"{ds['detection_rate']:.1%}",
                          "recall on test", tone="green"), unsafe_allow_html=True)
    b4.markdown(stat_card("False-alarm rate", f"{ds['false_alarm_rate']:.2%}",
                          f"{ds['fp']:,} legit flagged"), unsafe_allow_html=True)

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
        fit = r.get("Fit", "—")
        icon = {"good": "🟢", "slight overfit": "🟡", "overfit": "🔴", "underfit": "🟠"}.get(fit, "⚪")
        st.markdown(f"{icon} **Fit:** {fit} · train AUC {r.get('Train AUC')} vs test AUC "
                    f"{r.get('ROC-AUC')} — _{r.get('Fit reason', '')}_")
        st.plotly_chart(viz.confusion(r["confusion_matrix"], f"{name} · confusion matrix"),
                        width="stretch")

apply_scroll()
