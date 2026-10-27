
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate whether "
    "the customer may leave the company."
)

st.info("This prediction is an estimate, not a guarantee.")

# Customer information form
with st.form("customer_form"):

    st.subheader("Customer Information")

    gender = st.selectbox(
        "Gender", ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Has Partner?", ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Has Dependents?", ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (months)", 0, 72, 12
    )

    phone = st.selectbox(
        "Phone Service", ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0
    )

    submitted = st.form_submit_button(
        "Predict Customer Churn"
    )

# Predict when the form is submitted
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Calculate churn probability
    probability = model.predict_proba(customer)[0, 1]
    prediction = model.predict(customer)[0]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated churn probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.warning(
            "The model predicts that this customer "
            "may churn."
        )
    else:
        st.success(
            "The model predicts that this customer "
            "may stay."
        )

    st.caption(
        "Predictions depend on the training data "
        "and selected model."
    )
