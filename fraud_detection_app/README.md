# 🛡️ Sentinel — AI Fraud Detection Studio

A premium, multi-page Streamlit application for fraud detection that works **with
or without labels**, **explains every decision** in plain English, and shows
**business impact in dollars** — wrapped in a deep-navy + gold glass-morphism UI.

> Sentinel is the next-generation rebuild of this repo's original single-file app
> (which still lives at the repo root). It lives under `fraud_detection_app/` and
> can be developed independently.

---

## ✨ Highlights

- **Supervised _and_ unsupervised** — auto-detects whether your data has a label
  column. No labels? It switches to anomaly detection (Isolation Forest, LOF,
  One-Class SVM, Elliptic Envelope, optional Autoencoder) and gives every row a
  0–100 risk score with reasons.
- **Explainability first** — SHAP (with graceful fallback to native/permutation
  importance), per-prediction risk factors, a live **what-if** slider panel, and
  plain-English summaries of every prediction and metric.
- **Business impact** — translates the confusion matrix into fraud caught,
  fraud missed, false-alarm cost, net savings, and ROI.
- **Leakage-free & reproducible ML** — train/validation/test splits, threshold
  tuned on validation and scored on test, imbalance corrected exactly once
  (SMOTE *or* class weights), every model seeded.
- **Premium UI** — glass cards, gold accents, interactive Plotly charts, pipeline
  progress indicator.
- **Graceful degradation** — heavy AutoML/DL libraries are all optional and
  lazy-loaded; the app runs fully on the small core stack.

---

## 📁 Structure

```
fraud_detection_app/
├── app.py                       # Home / dashboard (Streamlit entry point)
├── pages/
│   ├── 1_📊_Data_Upload.py
│   ├── 2_🔍_EDA.py
│   ├── 3_⚙️_Preprocessing.py
│   ├── 4_📈_Model_Training.py   # supervised compare + unsupervised anomaly
│   ├── 5_🎯_Prediction.py       # single + batch, supervised + unsupervised
│   ├── 6_📚_Model_Explainability.py
│   └── 7_📊_Dashboard.py        # summary + fraud-pattern library + learning center
├── utils/
│   ├── __init__.py
│   ├── config.py                # theme, CSS, session state, UI atoms
│   ├── data_processor.py        # auto-detect, feature engineering, preprocessing
│   ├── model_trainer.py         # registry, get_proba, evaluation, AutoML, $ impact
│   ├── unsupervised.py          # anomaly detectors, risk scoring, explanations
│   ├── model_explainer.py       # SHAP / importance / risk factors / glossary
│   └── visualizer.py            # Plotly charts
├── assets/style.css             # glass-morphism theme
├── models/                      # saved model + scaler + features (auto-created)
├── requirements.txt
├── .env.example
├── README.md
└── CHANGELOG.md
```

---

## 🚀 Quick start

```bash
cd fraud_detection_app
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. No data? Click **“Generate synthetic PaySim
sample”** on the Home page to explore everything instantly.

---

## 🗺️ Workflow

1. **Data Upload** — drop a CSV (or generate a sample). Sentinel detects the
   target column, amount column, types, and fraud rate, and shows a data-quality
   report. Confirm/override the target; choosing “(none)” enables unsupervised mode.
2. **EDA** — auto-generated insights, interactive distributions, correlation,
   fraud-by-category, and time trends.
3. **Preprocessing** — automatic / manual / hybrid feature modes, scaler choice,
   correlation removal, SMOTE. Produces leakage-free train/val/test splits.
4. **Model Training** — pick classic models + ensembles (+ optional AutoML), set a
   cost model, train, and compare via table, ROC/PR curves, radar, confusion
   matrices, and a **business-impact** scorecard. Best model is saved automatically.
5. **Prediction** — single transaction form or batch CSV; supervised or
   unsupervised; each result comes with a risk gauge and the top contributing factors.
6. **Explainability** — global SHAP/importance, what-if sliders, plain-English
   reasons.
7. **Dashboard** — executive summary, drift snapshot, fraud-pattern library, and a
   plain-English learning center / glossary.

---

## 🔌 Optional integrations

The app runs on the core stack. To enable extras, install them (ideally in an
isolated env to avoid version conflicts):

| Feature | Install | Notes |
|---|---|---|
| Fast AutoML | `pip install flaml` | Lightest; recommended first AutoML option |
| Enterprise AutoML | `pip install h2o` | Requires Java |
| SOTA AutoML | `pip install autogluon.tabular` | Large download |
| Genetic AutoML | `pip install tpot` | Slow; pins older sklearn |
| Deep autoencoder | `pip install tensorflow` | Adds the Autoencoder detector |
| Richer explanations | `pip install shap` | Already in core; enables SHAP plots |

> ⚠️ **AutoML models are not saved to disk** (H2O/AutoGluon/FLAML aren't
> joblib-safe). If an AutoML model wins, it's used for the current session and a
> warning is shown.

---

## 🚢 Deployment

- **Streamlit Cloud** — point it at `fraud_detection_app/app.py`.
- **Docker** — `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **.env** — copy `.env.example` → `.env` for AutoML budgets and (placeholder)
  alert webhooks.

---

## 📄 License

MIT.
