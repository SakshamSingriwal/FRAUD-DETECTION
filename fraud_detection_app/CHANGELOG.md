# Changelog — Sentinel

## v3.0 — New multi-page platform

First release of **Sentinel**, a ground-up rebuild of the original single-file
fraud app into a modular, multi-page Streamlit platform.

### Added
- **Multi-page app** (`app.py` + 7 pages) with a premium glass-morphism theme
  (deep navy + gold), pipeline progress indicator, and reusable UI atoms.
- **Auto-detection** of target column, amount column, feature types, fraud rate,
  and PaySim schema (`data_processor.detect_metadata`).
- **Unsupervised mode** — when no label is detected, anomaly detection with
  Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, and an optional
  TensorFlow autoencoder; 0–100 risk scoring, smart auto-thresholding, and
  per-row plain-English "why it's unusual" explanations.
- **Explainability** — SHAP (with graceful fallback to native / permutation
  importance), per-prediction top risk factors, live **what-if** sliders, and a
  plain-English metric glossary + learning center.
- **Business impact** — fraud caught / missed, false-alarm cost, net savings, ROI.
- **Synthetic data generator** so the app is usable with zero setup.
- **Interactive Plotly** charts throughout (ROC, PR, radar, gauges, confusion,
  anomaly PCA scatter, SHAP summary, what-if waterfall).
- **AutoML** wrappers for FLAML / H2O / AutoGluon — lazy-loaded, optional.
- **Model persistence** with an explicit guard: AutoML leaders are not pickled
  (`ModelNotPersistableError`) and kept session-only.
- **Data-drift snapshot** and a **fraud-pattern library** on the dashboard.

### Correctness (carried over from the v2.1 production hardening)
- Single `get_proba()` that **raises** instead of returning silent zeros.
- Isolation Forest wrapped with **train-time** score scaling (single-row works).
- Threshold tuned on a **validation** split, metrics reported on the **test** split.
- Imbalance corrected **once** — SMOTE *or* class weights, never both.
- Correlation removal, scaling, and SMOTE all learn from the **train split only**.
- Every stochastic model seeded with `RANDOM_STATE=42`.
- Defensive NaN/column handling so prediction never crashes on messy input.

### Notes / non-goals
- The original root-level `app.py` / `utils.py` are **untouched** and still work.
- Email/Slack alerting and PDF/PPTX export are scaffolded (`.env.example`,
  CSV/Excel download) but not fully wired — they need deployment-specific creds.
- The full optional dependency set (TensorFlow + Torch + DGL + H2O + AutoGluon +
  TPOT + PyCaret) cannot coexist in one environment; extras are documented and
  lazy-loaded individually.

### Testing plan (manual)
Run `streamlit run app.py`, then verify:
1. **Synthetic data** — Home → generate sample → loads and detects supervised mode.
2. **Supervised** — upload labeled CSV → Preprocessing → train RF + LR → comparison
   table, ROC/PR/radar, and business-impact scorecard render; best model saved.
3. **Unsupervised** — set target to “(none)” → Preprocessing → run Isolation Forest
   → anomaly scatter + risk distribution; single prediction returns a **non-zero,
   varying** risk.
4. **Reproducibility** — train the same models twice → identical metrics & best model.
5. **Single & batch prediction** — both modes return scores; batch CSV downloads.
6. **Explainability** — global importance renders; what-if sliders move the gauge.
7. **AutoML** (if installed) — scores non-zero; warns that it isn't saved to disk.
8. **Edge cases** — single-class target raises a clear message; messy CSV (missing
   columns / blanks) still scores.
