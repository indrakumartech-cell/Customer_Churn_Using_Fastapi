# fastapi-project/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_churn

app = FastAPI(title="Customer Churn API with SHAP", version="1.0")

class CustomerInput(BaseModel):
    monthly_charges: float
    tenure_months: int
    total_support_calls: int

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}

@app.post("/predict")
def get_churn_prediction(data: CustomerInput):
    result = predict_churn(data.monthly_charges, data.tenure_months, data.total_support_calls)
    return result