# 🛡️ Real-Time Fraud Detection & Risk Scoring System

An intelligent, **hybrid anomaly detection and supervised learning system** that evaluates the risk level of financial transactions in real time.
It combines **unsupervised anomaly detection (Isolation Forest)** and **supervised fraud classification (XGBoost)** to deliver fast, explainable, and high-precision fraud predictions.

---

## 🚀 Overview

This project demonstrates a **complete end-to-end machine learning system**, including:

* Real-time fraud risk scoring via **Streamlit dashboard**
* Dual-layer detection: **Isolation Forest (unsupervised)** + **XGBoost (supervised)**
* Live model loading via remote URLs (supports large `.pkl` models)
* Session-based transaction simulation and decision logic
* Batch CSV evaluation with downloadable results
* Container-ready deployment for **Render**, **Azure**, or **Docker Compose**

---

## 🧰 Tech Stack

| Category                    | Tools                                   |
| --------------------------- | --------------------------------------- |
| **Frontend (Dashboard)**    | Streamlit                               |
| **Machine Learning**        | XGBoost, IsolationForest (Scikit-learn) |
| **Feature Engineering**     | Custom pipeline (`FeatureEngineer`)     |
| **Model Storage & Serving** | Azure Blob Storage / Remote URLs        |
| **Deployment**              | Docker, Render                          |
| **Task Queue (optional)**   | Celery, Redis                           |
| **Monitoring (optional)**   | Flower                                  |
| **Environment & Config**    | dotenv, joblib, pandas, numpy           |

---

## ⚙️ Workflow & Architecture

### 🔹 1. Data Preprocessing

* Transactional data is standardized and transformed using the `FeatureEngineer` class.
* Computes ratios and behavioral indicators:

  * `amountToAvgVolumeRatio`
  * `isFirstTransaction`
  * Time-based and account activity features
* Ensures **no data leakage** between training and prediction.

### 🔹 2. Unsupervised Anomaly Detection (Isolation Forest)

* Detects unusual transaction behavior patterns.
* Produces an **anomaly score** (`score_shifted`), adjusted using the optimal learned threshold (`BEST_THRESH`).

### 🔹 3. Supervised Classification (XGBoost)

* Consumes both raw and anomaly-based features.
* Produces a **fraud probability score (`Risk Score`)**.
* Decisions are tiered:

| Risk Probability | Decision  | Action                        |
| ---------------- | --------- | ----------------------------- |
| ≥ `T_HIGH`       | ❌ BLOCK   | Auto-reject transaction       |
| ≥ `T_LOW`        | 🟡 REVIEW | Flag for manual investigation |
| < `T_LOW`        | ✅ ALLOW   | Safe to process               |

### 🔹 4. Streamlit UI

* Real-time interface for **single transaction simulation** and **batch risk evaluation**.
* Displays transaction history per session.
* Supports download of results as `.csv`.

---

## 🧠 Model Logic

**Hybrid decision rule:**

```python
if RiskScore >= T_HIGH:
    Decision = "BLOCK"
elif RiskScore >= T_LOW:
    Decision = "REVIEW"
else:
    Decision = "ALLOW"
```

This combines anomaly-derived thresholds (`score_shifted`) with supervised confidence from `xgb.predict_proba`.

---

## 📈 Model Performance

| Model                               | Precision | Recall   | F1-score | ROC-AUC  |
| ----------------------------------- | --------- | -------- | -------- | -------- |
| **Isolation Forest (unsupervised)** | 0.24      | 0.49     | 0.32     | -        |
| **XGBoost (supervised)**            | **0.64**  | **0.77** | **0.70** | **0.98** |

✅ **High precision and recall balance** even with severe class imbalance.
✅ XGBoost correctly identifies **>70% of fraudulent cases** while minimizing false positives.
✅ Isolation Forest provides behavioral context that improves supervised accuracy.

---

## 🧩 Streamlit Application

### 🔹 Features

* **Single Transaction Mode**

  * Simulate user behavior (transfer, cash-out, etc.)
  * Real-time fraud decision and sender balance update
  * Session-based customer memory and transaction log

* **Batch Upload Mode**

  * Upload `.csv` of transactions
  * Automated risk classification with review summary
  * Download results instantly

* **Model Loading via URLs**

  * Automatically fetches and loads large `.pkl` files (e.g., 1.5 GB) from remote storage.
  * Cached with `@st.cache_resource` for speed.

---

## 📦 Project Structure

```plaintext
fraud-detection/
├── app/
│   ├── preprocess.py             # Feature engineering pipeline
│   ├── data_store.py             # Customer mock database
│   ├── main.py                   # Streamlit app entry point
│   └── feature_order.json        # Feature alignment
├── models/
│   ├── iso.pkl                   # Isolation Forest model (~1.5 GB compressed)
│   └── xgb.pkl                   # XGBoost model
├── .env                          # BEST_THRESH, T_LOW, T_HIGH, model URLs
├── Dockerfile
├── requirements.txt
├── README.md
└── output/
    └── shap_summary.png
```

---

## 🌐 Environment Variables

| Variable      | Description                        | Example               |
| ------------- | ---------------------------------- | --------------------- |
| `BEST_THRESH` | Best anomaly threshold             | `-0.0192`             |
| `T_LOW`       | Review threshold                   | `0.30`                |
| `T_HIGH`      | Block threshold                    | `0.85`                |
| `ISO_URL`     | Remote Isolation Forest `.pkl` URL | `https://.../iso.pkl` |
| `XGB_URL`     | Remote XGBoost `.pkl` URL          | `https://.../xgb.pkl` |

---

## 🧪 How to Run

### **Option 1 — Streamlit (local)**

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Access the dashboard at:
👉 [http://localhost:8501](http://localhost:8501)

### **Option 2 — Docker (recommended)**

```bash
docker-compose up --build
```

### **Optional Services**

* Redis (for background tasks)
* Celery worker (for model retraining or async scoring)
* Flower dashboard (task monitoring)

---

## 🕹️ Example Workflow

### 🔸 Single Transaction

1. Choose a sender from dropdown.
2. Enter amount and transaction type.
3. Click **“Process Transaction”**.
4. View:

   * Model decision (`ALLOW`, `REVIEW`, `BLOCK`)
   * Risk probability
   * Updated sender balance
   * Logged transaction history

### 🔸 Batch Upload

1. Upload a `.csv` of transactions with expected columns.
2. View summary of classifications.
3. Download risk results with fraud probabilities.

---

## 🪶 Future Roadmap

* 🧩 Integrate **live transaction streaming (Kafka / WebSockets)**
* 🧠 Add more **temporal features**
* 🤖 Implement **model drift detection & auto-retraining**
* 🔒 Centralize model storage via **Azure Blob / AWS S3**
* 📊 Integrate **explainability dashboard (SHAP visualizer)**

---

## 👨‍💻 Author

**Oladosu Larinde**
Software Engineer | ML Engineer 

📫 **[LinkedIn](https://www.linkedin.com/in/olarindeladosu)**
📧 **[larindeakin@gmail.com](mailto:larindeakin@gmail.com)**
