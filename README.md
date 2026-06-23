# 🔍 Fraud Detection Studio — Streamlit App

A production-ready, end-to-end fraud detection pipeline wrapped in a clean Streamlit interface. Covers EDA → feature engineering → model training → threshold tuning → single/batch prediction.

---

## 📁 Folder Structure

```
fraud_detection_app/
├── app.py              # Main Streamlit application
├── utils.py            # All helper functions (feature engineering, training, evaluation, prediction)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── models/             # Auto-created; stores trained model, scaler, feature columns
```

---

## 🚀 Quick Start

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Optional AutoML libraries** (comment them back into requirements.txt if needed):
> ```bash
> pip install autogluon.tabular   # AutoGluon
> pip install h2o                  # H2O AutoML
> ```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🗺️ App Pages

| Page | What it does |
|------|-------------|
| **📊 Data Upload & EDA** | Upload your training CSV. See class distribution, fraud rates by type, correlation heatmap, and basic stats. |
| **🔧 Feature Engineering** | Automatically creates 15+ features, removes correlated ones, scales, and applies SMOTE. |
| **🤖 Model Training** | Select and train up to 11 models including XGBoost, LightGBM, CatBoost, Stacking Ensemble, and AutoML. Compare by ROC-AUC, F1, and cost. Downloads the best model. |
| **🎯 Single Prediction** | Fill a form with one transaction's details. Instantly see fraud probability and decision. |
| **📁 Batch Prediction** | Upload a new CSV. Get fraud scores, top 10 suspicious transactions, probability histogram, and a downloadable results file. |

---

## 📦 Expected CSV Columns (Training)

```
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud
```

> Compatible with the **PaySim** synthetic financial fraud dataset available on Kaggle.

### For Batch Prediction CSV
Same columns **except** `isFraud` is optional. If included, the app will compute live performance metrics.

---

## ⚙️ Engineered Features

| Feature | Description |
|---------|-------------|
| `error_balance_orig` | Balance error at origin account |
| `error_balance_dest` | Balance error at destination account |
| `is_origin_emptied` | 1 if origin account drained to zero |
| `dest_initial_zero` | 1 if destination balance started at zero |
| `amount_log` | Log-transformed transaction amount |
| `orig_balance_change` | Net change in origin balance |
| `dest_balance_change` | Net change in destination balance |
| `orig_balance_ratio` | Ratio of new to old origin balance |
| `dest_balance_ratio` | Ratio of new to old destination balance |
| `amount_to_orig_ratio` | Amount relative to origin balance |
| `is_merchant` | 1 if destination is a merchant (M…) |
| `is_cash_out_transfer` | 1 if type is CASH_OUT or TRANSFER |
| `type_*` | One-hot encoded transaction types |

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

## 💾 Saved Artefacts

After training, the following files are saved to `models/`:

- `<ModelName>.pkl` — The best trained model
- `scaler.pkl` — Fitted StandardScaler
- `feature_cols.pkl` — List of feature columns
- `best_model_name.pkl` — Name of the best model

Use the **"Load Saved Model"** button in the sidebar to restore them in future sessions.

---

## 🧠 Tips

- **Start with Random Forest + XGBoost** for fast, strong baselines.
- **SMOTE** is applied by default to handle class imbalance — toggle it off for large datasets to save memory.
- **Optimal Threshold** is tuned by minimising misclassification cost (FN cost >> FP cost for fraud).
- For large datasets (>1M rows), consider sampling before upload or using AutoGluon's time limit wisely.

---

## 📄 License

MIT — free to use, modify, and deploy.
