# fastapi-project/model.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap

# Sample synthetic data for customer churn
np.random.seed(42)
X = pd.DataFrame({
    'monthly_charges': np.random.uniform(20, 120, 100),
    'tenure_months': np.random.randint(1, 72, 100),
    'total_support_calls': np.random.randint(0, 10, 100)
})
y = (X['monthly_charges'] > 80) & (X['tenure_months'] < 12) | (X['total_support_calls'] > 5)
y = y.astype(int)

# Train XGBoost model
model = XGBClassifier()
model.fit(X, y)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

def predict_churn(monthly_charges: float, tenure_months: int, total_support_calls: int):
    input_data = pd.DataFrame([[monthly_charges, tenure_months, total_support_calls]], 
                              columns=['monthly_charges', 'tenure_months', 'total_support_calls'])
    
    # Prediction probability
    prob = float(model.predict_proba(input_data)[0][1])
    prediction = int(model.predict(input_data)[0])
    
    # SHAP values for explanation
    shap_vals = explainer.shap_values(input_data)
    
    return {
        "churn_prediction": prediction,
        "churn_probability": round(prob, 2),
        "explanation": {
            "monthly_charges_impact": float(shap_vals[0][0]) if len(shap_vals.shape) > 1 else float(shap_vals[0]),
            "tenure_impact": float(shap_vals[0][1]) if len(shap_vals.shape) > 1 else float(shap_vals[1]),
            "support_calls_impact": float(shap_vals[0][2]) if len(shap_vals.shape) > 1 else float(shap_vals[2])
        }
    }