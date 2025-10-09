import streamlit as st
import pandas as pd
import requests
import time
import os # 👈 New: Import the OS module

# -------------------------------
# ⚙️ Configuration
# -------------------------------
# 1. Get API base URL from environment variable, defaulting to localhost if not set
API_BASE_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000")

# -------------------------------
# 🎨 Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom Styling (Clean & Exotic)
# -------------------------------
st.markdown("""
    <style>
        /* General Body and Font */
        body {
            font-family: 'Inter', sans-serif;
        }

        /* Main Title - Clean Header */
        .main-title {
            text-align: left;
            font-size: 2.2rem;
            font-weight: 700;
            padding-bottom: 0.5rem;
            margin-bottom: 2rem;
        }

        /* Accent Color for Buttons/Components */
        :root {
            --primary-color: #6C2BEE; /* Royal Purple */
        }

        /* Primary Button Style */
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 10px;
            font-weight: 600;
            height: 3rem;
            width: 100%;
            border: none;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #4A00B9; /* Darker on hover */
        }

        /* Input Fields/Selectbox/NumberInput */
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {
            border-radius: 10px;
            border: 1px solid #ccc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            padding: 0.5rem;
        }
        
        /* Containers (Forms, Columns) for a Card Look */
        section[data-testid="stForm"] {
            padding: 2rem;
            border: 1px solid var(--accent-color);
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        /* Metric Styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: var(--primary-color);
        }
        
        /* Sidebar Navigation Focus */
        .st-emotion-cache-1cypcdb { /* Target the radio button group container */
            border-radius: 10px;
            padding: 10px;
            background-color: #F8F4FF; /* Light background for contrast */
        }

        /* Dividers for cleaner separation */
        hr {
            border-top: 2px solid var(--accent-color);
        }

    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💳 Fraud Detection System</div>', unsafe_allow_html=True)
st.caption(f"Connected to API at: **{API_BASE_URL}**")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.header("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Predict Fraud", "Retrain Model", "Retrain Status"])

# -------------------------------
# Predict Fraud Page
# -------------------------------
if page == "Predict Fraud":
    st.header("⚡ Predict Fraudulent Transaction")
    st.markdown("""
    Fill in the transaction details below to get an **instant fraud risk assessment**.
    """)

    with st.form("predict_form"):
        # Use columns for a two-column layout in the form
        col_meta, col_type = st.columns(2)
        with col_meta:
            step = st.number_input("⏱ Step (1 step = 1 hour)", min_value=1, help="Represents the time step of the transaction. One step equals one hour.")
        with col_type:
            type_ = st.selectbox(
                "💳 Transaction Type",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
                help="Specifies the type of transaction being made."
            )
        amount = st.number_input("💰 Transaction Amount", min_value=0.0, help="The total amount being transferred or withdrawn.")

        st.divider()
        st.subheader("🧾 Sender (Origin Account)")
        
        col_orig_id, col_orig_bal = st.columns(2)
        with col_orig_id:
            nameOrig = st.text_input("Sender Account ID", placeholder="C123456789", help="Unique identifier for the sender’s account.")
        with col_orig_bal:
            oldbalanceOrg = st.number_input("Sender Balance (Before Transaction)", min_value=0.0, help="The sender’s balance before the transaction.")
        newbalanceOrig = st.number_input("Sender Balance (After Transaction)", min_value=0.0, help="The sender’s balance after the transaction.")

        st.divider()
        st.subheader("💼 Receiver (Destination Account)")
        
        col_dest_id, col_dest_bal = st.columns(2)
        with col_dest_id:
            nameDest = st.text_input("Receiver Account ID", placeholder="M987654321", help="Unique identifier for the receiver’s account.")
        with col_dest_bal:
            oldbalanceDest = st.number_input("Receiver Balance (Before Transaction)", min_value=0.0, help="Receiver’s balance before the transaction.")
        newbalanceDest = st.number_input("Receiver Balance (After Transaction)", min_value=0.0, help="Receiver’s balance after the transaction.")

        st.markdown("---") # Custom separator for the submit button
        submitted = st.form_submit_button("🚀 Predict Fraud Risk")

    if submitted:
        input_data = {
            "step": step,
            "type": type_,
            "amount": amount,
            "nameOrig": nameOrig,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "nameDest": nameDest,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }

        with st.spinner("Running fraud prediction..."):
            try:
                # API call to the /predict/ endpoint
                res = requests.post(f"{API_BASE_URL}/predict/", json=input_data)
                
                if res.status_code == 200:
                    result = res.json()
                    st.success("✅ Prediction completed successfully!")
                    
                    # Enhanced result display
                    col1, col2 = st.columns(2)
                    
                    # Highlight fraud prediction clearly
                    prediction_text = "🚨 **YES (Fraudulent)**" if result["prediction"] else "✅ **No (Legitimate)**"
                    prediction_color = "red" if result["prediction"] else "green"
                    
                    col1.markdown(f"**Fraudulent Transaction:** <span style='color:{prediction_color}; font-size: 1.2rem;'>{prediction_text}</span>", unsafe_allow_html=True)
                    
                    # Use metric for the probability
                    col2.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
                    
                else:
                    st.error(f"❌ Error during API call (Status {res.status_code}): {res.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"⚠️ Could not connect to API at `{API_BASE_URL}`. Please ensure the backend server is running.")
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred: {e}")

# -------------------------------
# Retrain Model Page
# -------------------------------
elif page == "Retrain Model":
    st.header("🧠 Retrain Model with New Data")
    st.info("Upload a new CSV file. The file should contain labeled data for retraining. This process runs asynchronously.")
    
    csv_file = st.file_uploader("Upload Training Data CSV", type=["csv"])

    if csv_file is not None:
        if st.button("🚀 Start Retraining"):
            with st.spinner("Uploading file and starting retraining job..."):
                try:
                    # Requests library handles multipart/form-data for file uploads
                    files = {"file": (csv_file.name, csv_file, "text/csv")}
                    res = requests.post(f"{API_BASE_URL}/retrain/", files=files)
                    
                    if res.status_code == 202: # 202 Accepted status for async job start
                        task_id = res.json().get("task_id")
                        st.success(f"✅ Retraining started successfully!")
                        st.markdown(f"**📍 Task ID:** `{task_id}`")
                        st.info("Navigate to 'Retrain Status' to track progress.")
                        st.session_state["last_task_id"] = task_id
                    else:
                        st.error(f"❌ Error during API call (Status {res.status_code}): {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"⚠️ Could not connect to API at `{API_BASE_URL}`.")
                except Exception as e:
                    st.error(f"Server error: {e}")

# -------------------------------
# Retraining Status Page
# -------------------------------
elif page == "Retrain Status":
    st.header("📊 Check Retraining Status")

    task_id_default = st.session_state.get("last_task_id", "")
    task_id = st.text_input("Enter Task ID", value=task_id_default)
    
    if "last_task_id" in st.session_state:
        st.caption(f"Quick check on last submitted task: `{st.session_state['last_task_id']}`")

    if st.button("🔄 Check Status"):
        if task_id:
            with st.spinner("Fetching retraining status..."):
                try:
                    # API call to the /retrain/status/{task_id} endpoint
                    res = requests.get(f"{API_BASE_URL}/retrain/status/{task_id}")
                    
                    if res.status_code == 200:
                        data = res.json()
                        status = data.get("status")
                        
                        st.markdown(f"**Current Status:** **`{status}`**")
                        
                        if status == "Completed":
                            st.balloons()
                            st.success("✅ Model retrained and deployed successfully!")
                            result = data.get("result", {})
                            
                            st.markdown("### Model Training Report")
                            st.write(f"• **Data Size:** **`{result.get('data_size', 'N/A')}`** samples")
                            st.write(f"• **Validation F1 Score:** **`{result.get('validation_score', 'N/A'):.4f}`**")
                            st.write(f"• **New Model Path:** `{result.get('model_path', 'N/A')}`")
                            
                        elif status == "Pending" or status == "Running":
                            st.info("⌛ Retraining is in progress. Please check back later.")
                            st.progress(data.get("progress", 0.1), text=f"Progress: {data.get('progress', 0.1)*100:.0f}%")
                            
                        elif status == "Failed":
                            st.error("❌ Retraining failed. Check the server logs for details.")
                            st.json(data)
                            
                        else:
                            st.warning("⚠️ Task status unknown or task not found.")

                    else:
                        st.error(f"❌ Error during API call (Status {res.status_code}): {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"⚠️ Could not connect to API at `{API_BASE_URL}`.")
                except Exception as e:
                    st.error(f"Server error: {e}")
        else:
            st.warning("⚠️ Please enter a task ID to check its status.")