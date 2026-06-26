# Changelog — Sentinel

## v4.1 — Session-only, recall-first, and a non-sticky Next button

### Changed / Removed
- **Removed run persistence** — the app no longer saves runs to disk; every time
  you open it you start a **fresh session** (Home → "Start new analysis"). The
  guided wizard (Previous / Next, locked stepper) stays. (`utils/runs.py` deleted.)
- **Recall-first decision objective** — Model Training now defaults to
  **"Catch all fraud (recall-first)"** with a recall-target slider (default 1.00).
  The threshold is set to catch your target share of frauds with the fewest false
  alarms, so **missed fraud (FN) → 0**, accepting more false positives — the right
  call when a missed fraud costs far more than reviewing a flagged legit txn.
  "Balanced (F1)" is still available.
- **Stable winner** — `pick_best_model` now ranks recall → precision → PR-AUC →
  LogLoss with rounded metrics, so the **same model wins on identical runs** (no
  more flip-flopping between near-tied models). The active model is also selectable.

### Fixed
- **Next button no longer "greys out" when scrolling** — it's always clickable and
  checks completion on click (showing a toast if the step isn't done), instead of a
  `disabled` flag that could go stale on a rerun.

## v4.0 — Runs, a guided wizard, and model selection

### Added
- **Runs** — every analysis is now a self-contained *run* (one full pipeline).
  The home page is a run manager: **create** a new run, **resume** any earlier run
  exactly where you left off, and **select + delete** runs to clean up. Each run is
  saved to disk under `runs/<id>/` (gitignored) and reloaded on demand
  (`utils/runs.py`).
- **Guided wizard navigation** — inside a run you move with **Previous / Next**
  only. The native page menu is hidden and the stage list is a locked stepper
  (current highlighted, done ✓, upcoming greyed). **Next** unlocks only when the
  stage is complete (data uploaded / preprocessing run / a model trained).
- **Active-model selection** — on Model Training you choose which trained model
  the **Prediction**, **Explainability**, and **Deploy** stages use (defaults to the
  best 🏆). The Dashboard adds a **Deploy → export model bundle** download.

### Notes
- Pipeline state autosaves at each step and on navigation, so a half-finished run
  is always resumable. AutoML (FLAML/AutoGluon) models are session-only and are
  not persisted in the snapshot (their metrics are kept).

## v3.7 — Fix empty "Why this prediction" bars

### Fixed
- The per-prediction **"Contribution to fraud score"** chart (Prediction &
  Explainability pages) showed feature names but **no bars**. Root cause: for
  CatBoost the SHAP path used `shap.Explainer(predict_proba, x)` with the single
  instance as its own background, collapsing every contribution to 0.
  - Now uses `shap.TreeExplainer` for tree models (CatBoost/LightGBM/XGBoost/RF/…),
    which needs no background and returns real values.
  - Linear models (Logistic Regression) use the exact `coef × value` contribution
    instead of perturbing a saturated sigmoid (which also yielded ~0).
  Verified: CatBoost top factor `orig_balance_change` (+1.97), etc. — bars render.

## v3.6 — Reliable AutoML set + auto-scroll to results

### Changed
- **Trimmed AutoML to what reliably works:** kept **FLAML + AutoGluon**, removed
  **TPOT** (flaky on Windows via its dask backend) and **H2O AutoML** (needs a Java
  runtime). Classic ML, ensembles, and unsupervised detectors are unchanged.
- **Auto-scroll to results:** after clicking **Run preprocessing**, **Train**, or
  **Predict**, the page now smooth-scrolls to the freshly rendered results so the
  output is never left off-screen. Implemented with a small `anchor()` /
  `request_scroll()` / `apply_scroll()` helper in `config.py`.

## v3.5 — AutoML/SHAP availability fixes + TPOT support

### Fixed / Added
- **AutoML & SHAP showed as "skipped" even when installed** — the real cause is an
  environment mismatch (the app must run in the Python where the optional libraries
  live). No code bug, but two real improvements:
  - AutoML failures now report the **actual reason** (not installed / needs Java /
    error message) instead of a vague "unavailable or failed — skipped".
  - Added **TPOT** as a first-class AutoML option (it was never wired in). The branch
    is version-robust (classic `scoring`/`generations` *and* TPOT 1.x `scorers`) and
    hands TPOT 1.x a **threaded dask client** so it runs on Windows (avoids the
    "Nanny failed to start" / "No clients found" errors).
- Note: `H2O AutoML` needs a Java runtime; TPOT 1.x needs `setuptools<81`.

## v3.4 — Premium UI refresh + clearer data profile

### Added / Changed
- **Major UI polish:** premium fonts (Plus Jakarta Sans + Inter), an animated aurora
  background, glowing glass cards with hover-lift and sheen, a shimmering hero,
  redesigned stat cards (now with icons and a top accent glow), polished buttons
  (shine sweep), pill tabs with a glowing indicator, gold-accented inputs/sliders,
  sidebar nav active-state highlight, custom scrollbar, and animated verdict banners.
- **Summary statistics → column profile:** replaced the raw `df.describe().T` (whose
  `count` looked identical for every column and confused users) with a readable
  profile showing dtype, **non-null count, missing %**, unique count, and key stats —
  plus a caption explaining that identical non-null counts simply mean complete data.

## v3.3 — Auto-selected scaler + Select-all fix

### Added
- **Automatic scaler selection** (`recommend_scaler`): the Preprocessing page now
  defaults to **Auto**, profiling the engineered features and choosing
  **RobustScaler** when data is heavily skewed / outlier-heavy (money amounts,
  balances) and **StandardScaler** otherwise. The chosen scaler and the reason are
  shown; standard / robust / minmax remain available as manual overrides.
  (Verified: real & synthetic PaySim → robust; Gaussian data → standard.)

### Fixed
- **Best-model selection** no longer crowns whichever model edges out ROC-AUC by a
  noise-level margin. On near-separable data the top models are statistically tied
  (~0.998 AUC), and a single Decision Tree was winning by 0.0004 despite coarser
  probabilities and weaker PR-AUC. `pick_best_model` now chooses the best **average
  rank across PR-AUC, F1, ROC-AUC and LogLoss** — a well-rounded, well-calibrated
  winner (CatBoost on the sample data). The comparison table now sorts by PR-AUC.
- **"Select all classic models"** on the Model Training page did nothing — keyed
  checkboxes ignore their `value=` arg on rerun. Now uses an `on_change` callback
  that sets each model checkbox's state (verified: selects all 10 and trains them).
- Added `.claude/` (Claude Code local settings) to `.gitignore`.

## v3.2 — Debug pass, simpler UX, and a full term glossary

### Changed
- **Removed the amount-column selector** (Data Upload) and all `amount_col`
  plumbing in `app.py`, `config.py`, and `data_processor.py` — it drove only the
  removed dollar/ROI feature and added noise for students.
- **Migrated `use_container_width=True` → `width="stretch"`** across all pages and
  the visualizer (Streamlit deprecated the old arg) — removes warnings, future-proof.

### Added — education / explainability
- **3-part metric glossary** (`METRIC_HELP`): every score now has *what it means*,
  *why it matters*, and *what a good value looks like* — including ROC-AUC, PR-AUC,
  Recall, Precision, F1, LogLoss, Train AUC, Fit, and Threshold.
- **Inline "What does each metric mean?"** expander on the Model Training page, so the
  definitions sit right next to the comparison table (TP/FP/FN/TN explained too).
- Learning Center renders the full 3-part glossary.

### Verified — full debug pass (both datasets)
Exercised against the real `Fraud_Analysis_Dataset.csv` (11,142 rows) **and** the
synthetic generator, via a temporary `streamlit.testing` harness (removed before
commit). **32/32 checks passed:**
- detect → preprocess (automatic + manual) → train every classic model
  (LR, DT, RF, XGBoost, LightGBM, Isolation Forest, Stacking, Voting) with no errors
  and a fit verdict each;
- detection summary, PSI drift (target excluded), single + batch prediction
  (probabilities in [0,1]), global + local explainability, all four unsupervised
  detectors, and every Plotly chart;
- all 8 pages render with no exception;
- interactive paths click-tested: preprocessing run, supervised training, unsupervised
  detection, and single prediction.

## v3.1 — Simpler, more rigorous training & honest monitoring

Senior-data-scientist pass focused on correctness over features.

### Changed
- **Removed the cost model.** Threshold tuning no longer needs FN/FP/review-cost
  inputs. The decision threshold now **maximises F1 on the validation split**
  (`best_threshold`), reported on the untouched test split — fewer assumptions,
  always well-defined.
- **Replaced dollar "business impact" / ROI** (which relied on a fabricated average
  amount) with an honest **detection summary**: fraud caught/missed, detection rate
  (recall), false-alarm rate, and precision — all directly observed on the test set.

### Fixed
- **Data-drift snapshot was statistically invalid** and produced false alarms
  (e.g. "drift in isFraud, step, amount"). Root causes: it included the **label**,
  split by **arbitrary row order**, and used **mean-shift %** (undefined for
  zero-heavy/count columns). Replaced with **PSI (Population Stability Index)** on
  binned distributions, computed on **features only** (target + ID columns excluded),
  **ordered by a time column** (`step`/datetime) when present, with standard
  thresholds (<0.10 stable · 0.10–0.25 moderate · >0.25 significant).

### Added — under/over-fitting control
- **Regularised model defaults** (shallow trees, leaf-size floors, row/column
  subsampling, L2) to keep variance in check.
- **Early stopping on the validation set** for XGBoost / LightGBM / CatBoost
  (`_fit`), defensive across library versions.
- **Fit diagnosis** per model (`_fit_diagnosis`): labels each model
  **underfit / good / slight overfit / overfit** from the train→test ROC-AUC gap,
  surfaced in the comparison table, per-model details, and dashboard scorecard.
- Appropriate **loss/eval metrics** for imbalanced data (logloss/AUC, PR-AUC reported).

### Improved
- **Synthetic generator** now injects class overlap + label noise, so demo data is
  realistic (AUC well below 1.0) and the fit/curve diagnostics are meaningful.

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
