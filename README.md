# 🛡️ Fraud Detection Studio v2.1

A production-ready, end-to-end fraud detection pipeline wrapped in a clean dark Streamlit UI.  
Covers **EDA → target selection → feature engineering (auto or manual) → model training → threshold tuning → single/batch prediction**.

> **v2.1** fixes five correctness bugs that silently produced wrong results
> (see [`CHANGELOG.md`](CHANGELOG.md)). Notably: the decision threshold is now
> tuned on a dedicated **validation split** and reported on an untouched **test
> split**; class imbalance is corrected **once** (SMOTE *or* class weights);
> Isolation Forest single-row scoring works; AutoML models score correctly; and
> every model is seeded for reproducibility.

---

## 📁 Folder Structure

```
fraud_detection_app/
├── app.py              # Main Streamlit application
├── utils.py            # Helper functions (feature engineering, training, prediction)
├── requirements.txt    # Python dependencies
├── CHANGELOG.md        # What changed in v2.0
├── README.md           # This file
└── models/             # Auto-created; stores trained model, scaler, feature columns, config
```

---

## 🚀 Quick Start

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## 🗺️ App Pages

| Page | What it does |
|------|-------------|
| **📊 Data Upload & EDA** | Upload CSV. **Select target column.** Choose Automatic or Manual feature mode. Explore class distribution, correlation heatmap, transaction types. |
| **🔧 Feature Engineering** | Configure test size, SMOTE, correlation threshold and run the preprocessing pipeline. |
| **🤖 Model Training** | Select and train up to 11 models. Compare by ROC-AUC, F1, and cost. Download the best model. |
| **🎯 Single Prediction** | Dynamic form matching your feature mode. Animated FRAUD / LEGIT result card. |
| **📁 Batch Prediction** | Upload a CSV. Get fraud scores, top 10 suspicious rows, probability histogram, download results. |

---

## ⚙️ New in v2.0: User-Selectable Target & Feature Mode

### Target Column Selection
On the **Data Upload** page, pick any binary column as your prediction target.  
`isFraud` is pre-selected if present; otherwise the app suggests binary numeric columns.

### Feature Selection Mode

| Mode | How it works |
|------|-------------|
| **Automatic** (default) | Runs `engineer_features()` — 15+ domain-specific features for PaySim-schema data. Recommended for the included dataset. |
| **Manual** | You pick which raw columns to use. Categoricals are one-hot encoded automatically. Scaling and SMOTE still apply. |

Both modes support optional correlation removal and SMOTE balancing.

---

## 📦 Expected CSV Columns

### Automatic Mode (PaySim schema)
```
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, <target_col>
```

### Manual Mode
Any CSV with your chosen feature columns + target column.

### Batch Prediction
Same schema as training (target column optional — if present, live metrics are shown).

---

## 🤖 Supported Models

| Model | Library |
|-------|---------|
| Logistic Regression | scikit-learn |
| Decision Tree | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| CatBoost | catboost |
| Isolation Forest | scikit-learn |
| Stacking Ensemble | scikit-learn |
| H2O AutoML | h2o *(optional)* |
| AutoGluon | autogluon *(optional)* |

---

## 💾 Saved Artefacts (`models/`)

| File | Contents |
|------|----------|
| `<ModelName>.pkl` | Best trained model |
| `scaler.pkl` | Fitted StandardScaler |
| `feature_cols.pkl` | Final feature column list |
| `best_model_name.pkl` | Name of the best model |
| `config.pkl` | `target_col` and `feature_mode` used during training |

Use **"Load Saved Model"** in the sidebar to restore them across sessions.

> ⚠️ **AutoML models (H2O / AutoGluon) are not saved to disk** — they are not
> joblib-safe and live in their own runtime (JVM cluster / AutoGluon dir). If an
> AutoML model wins, it is used for the current session only and the app shows a
> warning. Pick a standard model if you need a persistable artefact.

---

## 🧠 Tips

- **Start with Random Forest + XGBoost** for fast, strong baselines.
- **Automatic mode** is recommended for PaySim data — it captures balance-error and drain patterns that are strong fraud signals.
- **Manual mode** is ideal when you have a custom dataset with pre-engineered features.
- **SMOTE** handles class imbalance by default — disable it for very large datasets.
- **Optimal Threshold** minimises FN-weighted misclassification cost (FN >> FP for fraud).

---

## 📄 License

MIT — free to use, modify, and deploy.