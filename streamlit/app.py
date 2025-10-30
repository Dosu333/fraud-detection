import os
import json
import joblib
import requests
import pandas as pd
from dotenv import load_dotenv
from preprocess import FeatureEngineer
from data_store import get_initial_customer_db
import streamlit as st
from io import BytesIO


load_dotenv()

BEST_THRESH = float(os.getenv("BEST_THRESH", "-0.0192"))
T_LOW = float(os.getenv("T_LOW", "0.30"))
T_HIGH = float(os.getenv("T_HIGH", "0.85"))

ISO_URL = os.getenv("ISO_URL")
XGB_URL = os.getenv("XGB_URL")

EXPECTED_COLUMNS = [
    "timestamp",
    "type",
    "amount",
    "oldbalanceOrg",
    "avgDailyVolumeSoFar",
    "avgDailyVolumeBeforeTxn",
    "amountToAvgVolumeRatio",
    "isFirstTransaction",
]


@st.cache_resource
def load_models():
    if not XGB_URL or not ISO_URL:
        raise ValueError("Model URLs are not set in environment variables.")
    iso_response = requests.get(ISO_URL)
    xgb_response = requests.get(XGB_URL)
    iso = iso_response
    xgb = xgb_response
    # iso = joblib.load(BytesIO(iso_response.content))
    # xgb = joblib.load(BytesIO(xgb_response.content))
    with open("feature_order.json") as f:
        feature_order = json.load(f)
    return iso, xgb, feature_order


def classify(df):
    iso, xgb, feature_order = load_models()
    X_iso = df[[c for c in feature_order if c != "score_shifted"]].fillna(0)
    anomaly = -iso.decision_function(X_iso)
    df["score_shifted"] = anomaly - BEST_THRESH

    X = df[feature_order].fillna(0)
    proba = xgb.predict_proba(X)[:, 1]
    df["Risk Score"] = proba

    def decision(p):
        if p >= T_HIGH:
            return "❌ BLOCK"
        if p >= T_LOW:
            return "🟡 REVIEW"
        return "✅ ALLOW"

    df["Decision"] = df["Risk Score"].apply(decision)
    return df


# ---------------- STREAMLIT UI ----------------
st.title("💳 Real-Time Transaction Risk Checker")

tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload"])

# -------- Single Transaction Mode --------
with tab1:
    st.subheader("Simulate a Single Transaction")

    # --- Initialize session state database ---
    if "CUSTOMER_DB" not in st.session_state:
        st.session_state.CUSTOMER_DB = get_initial_customer_db()

    if "TRANSACTION_HISTORY" not in st.session_state:
        st.session_state.TRANSACTION_HISTORY = []

    CUSTOMER_DB = st.session_state.CUSTOMER_DB
    HISTORY = st.session_state.TRANSACTION_HISTORY

    # --- Inputs ---
    sender_id = st.selectbox("Sender ID", options=list(CUSTOMER_DB.keys()))
    receiver_id = st.text_input("Receiver ID (optional)")
    txn_type = st.selectbox(
        "Transaction Type", ["TRANSFER", "PAYMENT", "CASH_IN", "CASH_OUT"]
    )
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    timestamp = pd.Timestamp.now()

    # --- Display sender balance ---
    sender_balance = CUSTOMER_DB[sender_id]["balance"]
    st.info(f"💰 Current balance for {sender_id}: {sender_balance:,.2f}")

    if st.button("🚀 Process Transaction"):
        sender_data = CUSTOMER_DB[sender_id]
        oldbalance = sender_data["balance"]
        avg_vol = sender_data["avgDailyVolumeSoFar"]
        is_first = sender_data["totalTransactions"] == 0

        # --- Validate sufficient balance ---
        if amount > oldbalance and txn_type != "CASH_IN":
            st.error("❌ Insufficient balance for this transaction.")
            st.stop()

        # --- Compute features ---
        data = {
            "timestamp": timestamp,
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": oldbalance,
            "avgDailyVolumeSoFar": avg_vol,
            "avgDailyVolumeBeforeTxn": avg_vol,
            "amountToAvgVolumeRatio": amount / (avg_vol + 1e-6),
            "isFirstTransaction": is_first,
        }

        single = pd.DataFrame([data])

        # --- Transform and classify ---
        features = FeatureEngineer().transform(single)
        result = classify(features)

        # --- Update balances if allowed (simulate transaction effect) ---
        decision = result["Decision"].iloc[0]
        risk_score = result["Risk Score"].iloc[0]

        if decision == "✅ ALLOW":
            # Deduct from sender if not cash-in
            if txn_type != "CASH_IN":
                CUSTOMER_DB[sender_id]["balance"] = oldbalance - amount
            else:
                CUSTOMER_DB[sender_id]["balance"] = oldbalance + amount

            # Increment transaction count
            CUSTOMER_DB[sender_id]["totalTransactions"] += 1

        # --- Log transaction ---
        HISTORY.append(
            {
                "timestamp": timestamp,
                "sender": sender_id,
                "receiver": receiver_id,
                "type": txn_type,
                "amount": amount,
                "oldbalanceOrg": oldbalance,
                "newBalance": CUSTOMER_DB[sender_id]["balance"],
                "Risk Score": risk_score,
                "Decision": decision,
            }
        )

        # --- Display results ---
        st.markdown(f"### 🧠 Model Decision: {decision}")
        st.metric("Predicted Fraud Probability", f"{risk_score:.2%}")

        with st.expander("🔍 Model Input"):
            st.write(single)

        with st.expander("🧾 Model Output"):
            st.write(result)

        with st.expander("💳 Updated Sender Info"):
            st.write(CUSTOMER_DB[sender_id])

    # --- Transaction History Section ---
    if HISTORY:
        st.markdown("### 📜 Transaction History (Session)")
        hist_df = pd.DataFrame(HISTORY)
        st.dataframe(hist_df.tail(10), use_container_width=True)

        st.download_button(
            "⬇️ Download Full History",
            pd.DataFrame(HISTORY).to_csv(index=False).encode("utf-8"),
            file_name="transaction_history.csv",
            mime="text/csv",
        )

# -------- Batch Upload Mode --------
with tab2:
    st.subheader("Upload CSV File of Transactions")

    # ---- Sample CSV Download ----
    sample_df = pd.DataFrame(
        [
            {
                "timestamp": "2025-10-30 12:00:00",
                "type": "TRANSFER",
                "amount": 1500.50,
                "oldbalanceOrg": 5000.00,
                "avgDailyVolumeSoFar": 2000.00,
                "avgDailyVolumeBeforeTxn": 1800.00,
                "amountToAvgVolumeRatio": 0.75,
                "isFirstTransaction": False,
            }
        ]
    )

    st.download_button(
        "📄 Download Sample CSV",
        sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )

    file = st.file_uploader("Select CSV File", type="csv")

    if file:
        # --- Keep original data intact ---
        original_df = pd.read_csv(file)
        df = original_df.copy()

        st.info("📊 Preview of Uploaded Data:")
        st.dataframe(df.head(), use_container_width=True)

        # Check for missing columns
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
        else:
            # --- Transform and classify ---
            transformed = FeatureEngineer().transform(df)
            classified = classify(transformed)

            # --- Attach results to original data ---
            results = original_df.copy()
            results["Risk Score"] = classified["Risk Score"]
            results["Decision"] = classified["Decision"]

            st.success("✅ Transactions evaluated!")

            # --- Decision summary ---
            st.markdown("### Decision Summary")
            st.write(results["Decision"].value_counts())

            # --- Detailed preview ---
            st.markdown("### Detailed Results Preview")
            st.dataframe(results.head(20), use_container_width=True)

            # --- Download full results ---
            st.download_button(
                "⬇️ Download Full Results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="risk_results.csv",
                mime="text/csv",
            )
