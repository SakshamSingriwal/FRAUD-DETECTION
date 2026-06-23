# Changelog

## v2.1 — Correctness & Production Hardening

This release fixes the five silent-wrong-output bugs identified in code review,
plus a set of related issues found during a full audit. No UI features were
removed; the public function signatures that changed are listed under
**Breaking changes** below.

---

### 🐞 The 5 critical bugs (from review)

**Bug 1 — AutoML models predicted everything as "legit"**
- *Cause:* `_get_proba()` returned all-zero probabilities for any model lacking
  `predict_proba` (H2O leader, etc.), so every prediction showed 0% fraud with
  no error.
- *Fix:* Introduced a single `get_proba(model, X)` entry point that explicitly
  handles **H2O** (`predict()` → `p1` column) and **AutoGluon**
  (`predict_proba` DataFrame → positive column), falls through to standard
  `predict_proba`, and **raises `ValueError`** if a model genuinely cannot
  score — it never returns silent zeros again.
- AutoML leaders are **not joblib-safe**, so `save_artefacts()` now raises
  `ModelNotPersistableError` for them and the app keeps them for the current
  session only (with a clear warning) instead of writing an unusable `.pkl`.
- `_get_proba` is retained as an alias of `get_proba` for backward
  compatibility.

**Bug 2 — Isolation Forest always returned 0 on a single prediction**
- *Cause:* the anomaly score was min-max scaled *within the current batch*; for
  one row `min == max`, so the score collapsed to 0 and fraud was never flagged.
- *Fix:* Added `IsolationForestWrapper`, which stores the score `min`/`max` from
  the **training set** at `fit()` time and reuses them for every later batch
  (including single rows). It exposes a standard `predict_proba`, so it flows
  through `get_proba` like any other model and pickles cleanly.

**Bug 3 — Threshold was tuned and scored on the same test set (leakage)**
- *Cause:* `evaluate_model()` picked the optimal threshold on the test set and
  then reported Precision/Recall/F1 on that same set → optimistic metrics.
- *Fix:* `preprocess()` now produces a **train / validation / test** split.
  `evaluate_model(model, X_val, y_val, X_test, y_test)` tunes the threshold on
  the validation split and reports all metrics on the untouched test split. The
  same discipline is applied to the AutoML wrappers.

**Bug 4 — Class imbalance corrected twice (SMOTE + class weights)**
- *Cause:* SMOTE balanced the data to ~50/50 *and* the models added
  `class_weight="balanced"` / `scale_pos_weight=10`, over-predicting fraud.
- *Fix:* `build_model_registry(use_class_weights=...)` makes weighting
  configurable. The training flow sets `use_class_weights = not apply_smote`, so
  imbalance is corrected **exactly once**: SMOTE *or* class weights, never both.

**Bug 5 — Results were not reproducible**
- *Cause:* Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, and
  the stacking sub-estimators had no seed.
- *Fix:* A global `RANDOM_STATE = 42` is now applied to **every** stochastic
  model (`random_state` / `random_seed`). Repeated runs give identical metrics
  and the same "best model".

---

### 🔍 Additional audit fixes

- **Correlation-removal leakage:** correlated features are now identified on the
  **training split only** (`find_correlated_features`) and the same columns are
  dropped from validation/test — previously the decision used the whole dataset.
- **Consistent missing-value handling:** `engineer_features()` and
  `prepare_input_for_prediction()` now coerce inputs to numeric and fill NaNs,
  so training and prediction handle messy/absent columns identically instead of
  crashing or silently skewing the scaler.
- **Robust AUC:** `roc_auc_score` is wrapped so a single-class slice yields
  `NaN` rather than raising mid-training.
- **Edge case — single-class target:** `preprocess()` raises a clear error when
  the target has only one class, instead of failing deep inside SMOTE/metrics.
- **Stacking / IsolationForest** now seeded and (for IsolationForest) wrapped so
  they persist and score correctly.

---

### ⚠️ Breaking changes (internal API only — no UI change)

- `preprocess()` returns extra keys: `X_val`, `y_val`, and `apply_smote`.
- `evaluate_model(model, X_val, y_val, X_test, y_test)` — now takes a validation
  set.
- `train_models(selected, X_train, y_train, X_val, y_val, X_test, y_test, use_class_weights=..., ...)`.
- `train_h2o_automl` / `train_autogluon` now take validation args.
- `build_model_registry(use_class_weights=False, ...)`.
- `remove_correlated_features()` replaced by `find_correlated_features()` (returns
  the drop list; the split-aware caller does the dropping).
- `get_proba()` is the new public name; `_get_proba` remains as an alias.

`app.py` was updated to use all of the above. Saved-model load
(`load_artefacts`) and the on-disk artefact format are **unchanged**, so
previously saved non-AutoML models still load.

---

### 🧪 Testing plan (manual)

Run `streamlit run app.py` and:

1. **Reproducibility (Bug 5):** Train the same model set twice (e.g. Random
   Forest + Logistic Regression). The ROC-AUC, threshold, and "best model" must
   be **identical** across runs.
2. **No double correction (Bug 4):** With SMOTE **on**, train Random Forest —
   precision should be healthy (not collapsed). Confirm via the comparison table
   that recall isn't ~1.0 with very low precision.
3. **Validation split (Bug 3):** On the Feature Engineering page, confirm the
   results show separate **Train / Validation / Test** sample counts and the
   caption "Threshold is tuned on the validation split…".
4. **Isolation Forest single prediction (Bug 2):** Train Isolation Forest, go to
   Single Prediction, and predict one row — the fraud probability must be a
   **non-zero, varying** value (not a flat 0.0000).
5. **AutoML scoring (Bug 1):** If H2O/AutoGluon is installed and wins, Single &
   Batch Prediction must show **non-zero** probabilities. Confirm a warning that
   the AutoML model is session-only (not saved to `models/`).
6. **No silent zeros (Bug 1):** Any model that cannot score raises a visible
   error in the per-model results instead of reporting 0% fraud.
7. **Batch with messy CSV:** Upload a batch CSV with a missing optional column /
   some blank cells — predictions still run (NaNs filled, columns aligned).
