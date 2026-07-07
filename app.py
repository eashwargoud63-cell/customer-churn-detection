
import streamlit as st
import joblib
import pandas as pd

model       = joblib.load("churn_model.pkl")
preprocessor = joblib.load("churn_preprocessor.pkl")

st.title("📡 Telco Customer Churn Predictor")
st.markdown("Enter customer details to predict whether they will churn.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender          = st.selectbox("Gender", ["Male", "Female"])
        senior          = st.selectbox("Senior Citizen", [0, 1])
        partner         = st.selectbox("Partner", ["Yes", "No"])
        dependents      = st.selectbox("Dependents", ["Yes", "No"])
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines  = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup    = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protect   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method   = st.selectbox("Payment Method", 
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges  = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)
        total_charges    = st.number_input("Total Charges ($)", min_value=0.0, value=800.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protect, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "ChargesPerMonth": total_charges / (tenure + 1),
        "ServiceCount": sum([phone_service == "Yes", online_security == "Yes",
                              online_backup == "Yes", device_protect == "Yes",
                              tech_support == "Yes", streaming_tv == "Yes",
                              streaming_movies == "Yes"]),
        "IsSeniorAlone": int(senior == 1 and dependents == "No")
    }])

    proc  = preprocessor.transform(input_df)
    pred  = model.predict(proc)[0]
    prob  = model.predict_proba(proc)[0][1]

    if pred == 1:
        st.error(f"⚠️ This customer is likely to CHURN  (Probability: {prob:.1%})")
    else:
        st.success(f"✅ This customer is likely to STAY  (Probability: {1 - prob:.1%})")
