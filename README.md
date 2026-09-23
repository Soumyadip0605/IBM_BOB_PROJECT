# 🔐 UPI Payment Fraud Detection

A full-stack data science project that detects fraudulent UPI (Unified Payments Interface) transactions using machine learning — with an interactive **Streamlit dashboard** as the frontend.

---

## 📂 Project Structure

```
upi_fraud_detection/
│
├── data/
│   └── upi_transactions.csv          ← Dataset (17 columns, ~N rows)
│
├── src/
│   ├── data_loader.py                ← Load, clean, encode raw CSV
│   ├── eda.py                        ← Plotly EDA charts + summary stats
│   ├── features.py                   ← Feature engineering + train/test split
│   ├── train.py                      ← Model training (LR, RF, GB, XGBoost)
│   └── evaluate.py                   ← ROC, PR, confusion matrix, feature importance
│
├── models/                           ← Saved .pkl model files (auto-created)
├── reports/                          ← Optional: exported charts / reports
│
├── dashboard.py                      ← Streamlit multi-page frontend
├── main.py                           ← CLI training pipeline entry point
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**File:** `data/upi_transactions.csv`  
**Source:** Synthetic UPI transaction data for fraud detection research.

| Column | Description |
|---|---|
| `transaction_id` | Unique transaction identifier |
| `timestamp` | Date and time of transaction |
| `transaction_type` | P2P, P2M, Bill Payment, Recharge |
| `merchant_category` | Food, Shopping, Fuel, Healthcare, etc. |
| `amount_inr` | Transaction amount in Indian Rupees |
| `transaction_status` | SUCCESS / FAILED / PENDING |
| `sender_age_group` | Age bracket of sender |
| `receiver_age_group` | Age bracket of receiver |
| `sender_state` | Indian state of sender |
| `sender_bank` / `receiver_bank` | Bank names |
| `device_type` | Android / iOS / Web |
| `network_type` | 4G / 5G / 3G / WiFi |
| `fraud_flag` | **Target** — 0 = Genuine, 1 = Fraud |
| `hour_of_day` | Hour extracted from timestamp (0–23) |
| `day_of_week` | Day name |
| `is_weekend` | 1 = weekend, 0 = weekday |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Frontend | **Streamlit** |
| Model persistence | pickle / joblib |

---

## ⚙️ Setup & Installation

### 1. Clone / download the project
```bash
cd upi_fraud_detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — CLI training only
```bash
python main.py
```
Optional flags:
```bash
python main.py --no-smote          # skip SMOTE
python main.py --test-size 0.25    # 25% test split
python main.py --seed 99           # custom random seed
```

### Option B — Launch the full dashboard
```bash
streamlit run dashboard.py
```
Then open **http://localhost:8501** in your browser.

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| 🏠 **Home** | KPI cards, class distribution, dataset preview |
| 📊 **EDA** | 10+ interactive charts across time, category, geography, correlation |
| 🤖 **Model Training** | Train LR / RF / Gradient Boosting / XGBoost in-browser |
| 📈 **Evaluation** | ROC, PR curves, confusion matrix, feature importance, threshold analysis |
| 🔍 **Live Predict** | Enter any transaction details → get fraud probability from all models |

---

## 🤖 ML Pipeline

```
Raw CSV
  └─▶ data_loader.py   (clean, type-cast)
        └─▶ features.py  (engineer: log_amount, same_bank, is_night, age_diff, OHE)
              └─▶ SMOTE oversampling  (handles ~10% fraud imbalance)
                    └─▶ Models trained in parallel:
                          • Logistic Regression  (baseline)
                          • Random Forest        (200 trees, class_weight=balanced)
                          • Gradient Boosting    (sklearn, 200 estimators)
                          • XGBoost              (scale_pos_weight for imbalance)
                    └─▶ evaluate.py  (ROC-AUC, Avg Precision, F1, CM, thresholds)
```

### Key engineered features
| Feature | Description |
|---|---|
| `log_amount` | log(1 + amount_inr) — normalises skewed distribution |
| `is_night` | 1 if transaction between 22:00–05:59 |
| `is_morning` | 1 if transaction between 06:00–10:59 |
| `same_bank` | 1 if sender & receiver use the same bank |
| `age_diff` | Ordinal distance between sender and receiver age groups |
| `month` / `quarter` | Seasonality features |

---

## 📈 Expected Performance (approximate)

| Model | ROC-AUC | F1 (Fraud) | Recall (Fraud) |
|---|---|---|---|
| Logistic Regression | ~0.70 | ~0.45 | ~0.60 |
| Random Forest | ~0.92 | ~0.72 | ~0.68 |
| Gradient Boosting | ~0.93 | ~0.74 | ~0.70 |
| XGBoost | ~0.94 | ~0.76 | ~0.72 |

> *Actual scores depend on the dataset split and SMOTE settings.*

---

## 📝 Notes

- The dataset is class-imbalanced (~10% fraud).  SMOTE is enabled by default during training.
- Models are saved as `.pkl` files in the `models/` directory after training.
- The **Live Predict** page requires models to be trained first (via the Model Training page or `main.py`).

---

## 👤 Author

Built as a data analysis project using the IBM Bob AI assistant.  
Dataset: `upi_transactions.csv`
