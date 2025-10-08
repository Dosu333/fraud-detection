# 🛡️ Online Transaction Fraud Detection System

A **machine learning–powered fraud detection pipeline** designed to identify fraudulent online financial transactions in real time.
The project uses supervised learning and model interpretability techniques to build a transparent, high-performing fraud detection system on a realistic, synthetic dataset.

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
* Addressed severe class imbalance using **SMOTE** and stratified sampling.

### 2. **Model Training**

Multiple supervised learning algorithms were tested:

* **Random Forest Classifier**
* **XGBoost Classifier** *(best performer)*

Models were trained to predict the `isFraud` label and evaluated on multiple metrics beyond accuracy.

### 3. **Hyperparameter Tuning & Validation**

* Optimized using `RandomizedSearchCV` and stratified cross-validation.
* Ensured model robustness and generalization on unseen data.

### 4. **Explainability with SHAP**

* **Global interpretability:** SHAP summary plots identify which features drive fraud predictions most.
* **Local interpretability:** Force and waterfall plots explain why *individual transactions* were flagged as fraud.

---

## ✅ Model Performance

| Metric                        | Score                                      |
| ----------------------------- | ------------------------------------------ |
| **ROC-AUC**                   | **0.983**                                  |
| **AUPRC (Average Precision)** | **0.8277**                                 |
| **Precision (fraud)**         | 0.65–0.83                                  |
| **Recall (fraud)**            | 0.71–0.79                                  |
| **Accuracy**                  | ~100% *(not relied upon due to imbalance)* |

🔍 **Interpretation:**
Despite extreme class imbalance (<0.2% fraud rate), the model achieves *top-tier precision-recall balance* and excellent separability (ROC-AUC ≈ 0.98).
SHAP analysis confirms that **transaction type**, **sender balance**, and **transaction amount** are the most predictive indicators of fraud.

---

## 🧠 Key Insights

* **Accuracy ≠ performance** — Precision, Recall, AUPRC, and ROC-AUC are the real indicators in fraud detection.
* **SHAP enhances trust** — Model interpretability is essential for explaining automated fraud flags.
* **Behavioral patterns matter** — Features capturing balance dynamics and transaction types are most influential.

---

## 🛠️ Future Improvements

* 🚀 **Deploy as an API** (FastAPI/Flask) for real-time scoring
* 🔁 **Stream monitoring** for live transaction data
* 🧩 **Temporal modeling** (e.g., sequence-based fraud detection using RNNs or Transformers)
* 🤖 **Hybrid approach** combining supervised + unsupervised anomaly detection
* 📈 **Model drift monitoring** to maintain long-term performance

---

## 📸 Sample SHAP Summary Plot

![SHAP Summary](outputs/shap_summary.png)

> Low sender balances and CASH_OUT/TRANSFER transaction types are the strongest indicators of fraud.

---

## 👨‍💻 Author

**Oladosu Larinde**
Lead Software Engineer | Machine Learning Enthusiast

📫 **Contact**

* [LinkedIn](https://www.linkedin.com/in/olarindeladosu)
* [Email](mailto:larindeakin@gmail.com)
