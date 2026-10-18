import streamlit as st
import requests

st.title("📊 Customer Churn Predictor with Explainable AI")
st.write("Yeh app predict karta hai ki koi customer company chhod kar jayega ya nahi, sath hi SHAP values se reason bhi batata hai.")

monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 70.0)
tenure_months = st.slider("Tenure (Months)", 1, 72, 12)
total_support_calls = st.slider("Total Support Calls", 0, 10, 2)

if st.button("Predict Churn"):
    payload = {
        "monthly_charges": monthly_charges,
        "tenure_months": tenure_months,
        "total_support_calls": total_support_calls
    }
    
    try:
        response = requests.post("https://customer-churn-using-fastapi.onrender.com/predict", json=payload)
        if response.status_code == 200:
            res = response.json()
            prob = res["churn_probability"] * 100
            
            if res["churn_prediction"] == 1:
                st.error(f"⚠️ Yeh customer churn kar sakta hai! Probability: {prob:.2f}%")
            else:
                st.success(f"✅ Yeh customer safe hai. Churn Probability: {prob:.2f}%")
                
            st.subheader("🔍 AI Explanation (SHAP Impact):")
            
            # JSON/Dictionary ko clean bullets mein print karne ke liye
            for key, value in res["explanation"].items():
                st.write(f"- **{key}**: {value}")
                
        else:
            st.error("Backend API se error aayi hai.")
    except Exception as e:
        st.error(f"FastAPI server se connect nahi ho paya: {e}")
