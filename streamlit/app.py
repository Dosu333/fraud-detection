import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from preprocess import FeatureEngineer
import streamlit as st


load_dotenv()

BEST_THRESH = float(os.getenv("BEST_THRESH", "-0.0192"))
T_LOW = float(os.getenv("T_LOW", "0.30"))
T_HIGH = float(os.getenv("T_HIGH", "0.85"))

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
    iso = joblib.load("iso.pkl")
    xgb = joblib.load("xgb.pkl")
    with open("feature_order.json") as f:
        feature_order = json.load(f)
    return iso, xgb, feature_order


iso, xgb, feature_order = load_models()


def classify(df):
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
    st.subheader("Enter transaction details")

    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    oldbalance = st.number_input(
        "Sender Balance Before Transaction", min_value=0.0, step=0.01
    )
    avgVol = st.number_input("Avg Daily Volume (Past Days)",
                             min_value=0.0, step=0.01)

    if st.button("Check Risk"):
        single = pd.DataFrame(
            [
                {
                    "amount": amount,
                    "oldbalanceOrg": oldbalance,
                    "avgDailyVolumeSoFar": avgVol,
                    "avgDailyVolumeBeforeTxn": avgVol,
                    "amountToAvgVolumeRatio": amount / (avgVol + 1e-6),
                }
            ]
        )

        single = FeatureEngineer().transform(single)
        result = classify(single)

        st.markdown(f"### Result: {result['Decision'].iloc[0]}")

        with st.expander("Advanced Details"):
            st.write(result)

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
