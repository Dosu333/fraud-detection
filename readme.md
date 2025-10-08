Perfect 👌 — here’s your **polished, recruiter-impressive** version of the README.
I’ve kept your authentic tone and structure but refined it to look like something a **Lead ML Engineer** or **ML Backend Engineer** would showcase in a portfolio or job application.

---

# 🛡️ Online Transaction Fraud Detection System

A **machine learning–powered fraud detection pipeline** that identifies fraudulent online financial transactions in real time.

This project demonstrates my ability to build a **complete end-to-end ML solution** — from data preprocessing and feature engineering to model training, explainability, and containerized deployment using FastAPI.

---

## 🧰 Tech Stack

* **Backend:** FastAPI, Uvicorn
* **ML Framework:** XGBoost, scikit-learn, pandas, numpy
* **Model Explainability:** SHAP
* **Serialization:** Joblib
* **Deployment:** Docker, Render
* **Version Control & CI/CD:** Git, GitHub Actions (ready for integration)

---

## 📊 Dataset Overview

The dataset simulates real-world financial transactions with the following key features:

| Feature                            | Description                                         |
| ---------------------------------- | --------------------------------------------------- |
| `step`                             | Time step (1 step = 1 hour)                         |
| `type`                             | Transaction type (TRANSFER, CASH_OUT, etc.)         |
| `amount`                           | Transaction amount                                  |
| `nameOrig`, `nameDest`             | Originator and recipient accounts                   |
| `oldbalanceOrg`, `newbalanceOrig`  | Sender’s balance before and after the transaction   |
| `oldbalanceDest`, `newbalanceDest` | Receiver’s balance before and after the transaction |
| `isFraud`                          | Target variable (1 = fraud, 0 = non-fraud)          |

**Engineered features:**

* `balanceDiffOrig`, `balanceDiffDest` — change in sender/receiver balances
* `orig_balance_ratio`, `dest_balance_ratio` — relative transaction amounts
* One-hot encoded transaction types (e.g., `type_CASH_OUT`, `type_TRANSFER`)

---

## ⚙️ Workflow & Methodology

### 1. **Data Preprocessing**

* Cleaned and validated raw transaction data.
* Engineered ratio-based and difference-based balance features.
* Encoded categorical features using one-hot encoding.
* Handled severe class imbalance using **SMOTE** and stratified sampling.

### 2. **Model Training**

Tested multiple supervised learning algorithms:

* Random Forest Classifier
* **XGBoost Classifier** *(selected for best performance)*

The final XGBoost model was trained to predict `isFraud` and evaluated using metrics optimized for imbalanced datasets.

### 3. **Hyperparameter Tuning**

* Tuned using `RandomizedSearchCV` and stratified cross-validation.
* Ensured robustness and generalization on unseen data.

### 4. **Explainability with SHAP**

* **Global interpretability:** SHAP summary plots show which features most influence fraud detection.
* **Local interpretability:** SHAP force and waterfall plots explain *why* individual transactions are flagged.

---

## ✅ Model Performance

| Metric                     | Score                                      |
| -------------------------- | ------------------------------------------ |
| **ROC-AUC**                | **0.983**                                  |
| **AUPRC (Avg. Precision)** | **0.8277**                                 |
| **Precision (fraud)**      | 0.65 – 0.83                                |
| **Recall (fraud)**         | 0.71 – 0.79                                |
| **Accuracy**               | ~100% *(not relied upon due to imbalance)* |

> Even with an extreme fraud rate (<0.2%), the model achieves high precision-recall balance and excellent separability (ROC-AUC ≈ 0.98).
> SHAP confirms that **transaction type**, **sender balance**, and **transaction amount** are the most predictive indicators of fraud.

---

## 🧠 Key Insights

* **Accuracy ≠ performance** — metrics like Precision, Recall, AUPRC, and ROC-AUC matter more in fraud detection.
* **Explainability builds trust** — SHAP enhances transparency for stakeholders.
* **Behavioral patterns dominate** — balance changes and transaction types carry the strongest fraud signals.

---

## 📦 Project Structure

```plaintext
fraud-detection/
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── endpoints.py           # API routes
│   ├── preprocess.py          # Feature engineering pipeline
│   ├── model/
│   │   └── fraud_model_pipeline_v1.pkl
│   └── __init__.py
├── notebooks/
│   ├── preprocessing.ipynb
│   └── models.ipynb
├── requirements.txt
├── Dockerfile
├── README.md
└── output/
    └── shap_summary.png
```

---

## 🕵️‍♂️ Fraud Detection API

A production-ready **FastAPI** microservice for real-time fraud prediction.

### 🚀 Features

* Real-time prediction endpoint
* Containerized with Docker
* Logs model loading and prediction performance
* Ready for CI/CD deployment on **Render**, **Azure**, or **AWS**

---

### 🧭 Running Locally

**Option 1 – Docker**

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

**Option 2 – Manual**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then visit 👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🧩 API Endpoints

#### `POST /predict/` — Predict fraud likelihood

**Request**

```json
{
  "step": 5,
  "type": "TRANSFER",
  "amount": 120000.50,
  "nameOrig": "C12345",
  "oldbalanceOrg": 50000.00,
  "newbalanceOrig": 0.00,
  "nameDest": "M67890",
  "oldbalanceDest": 0.00,
  "newbalanceDest": 120000.50
}
```

**Response**

```json
{
  "prediction": true,
  "fraud_probability": 88.82
}
```

---

## 🧠 Model Storage

For demonstration, the trained XGBoost pipeline is included in `app/model/fraud_model_pipeline_v1.pkl`.

In production, the model would be securely stored and versioned (e.g., in **Azure Blob Storage** or **AWS S3**) and automatically loaded during FastAPI startup.

---

## 🛠️ Future Improvements

* 🔁 **Stream monitoring** for live transaction feeds
* 🧩 **Temporal modeling** using RNNs or Transformers
* 🤖 **Hybrid fraud detection** (supervised + unsupervised)
* 📈 **Model drift monitoring** & retraining automation

---

## 📸 SHAP Summary Plot

![SHAP Summary](./output/shap_summary.png)

> Low sender balances and high-value TRANSFER/CASH_OUT transactions are the strongest indicators of fraud.

---

## 👨‍💻 Author

**Oladosu Larinde**
Lead Software Engineer | Machine Learning Enthusiast

📫 **Contact**

* [LinkedIn](https://www.linkedin.com/in/olarindeladosu)
* [Email](mailto:larindeakin@gmail.com)