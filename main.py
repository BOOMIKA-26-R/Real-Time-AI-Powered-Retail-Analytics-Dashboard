from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load Models and Data
churn_model = joblib.load("churn_model.pkl")
ts_model = joblib.load("sales_forecast.pkl")

@app.get("/")
def home():
    return {"status": "Retail API Live", "endpoints": ["/sales_forecast", "/churn_stats", "/total_sales"]}

# 1. KPI: Total Sales
@app.get("/total_sales")
def get_total_sales():
    items_df = pd.read_csv('data/olist_order_items_dataset.csv')
    total_val = round(items_df['price'].sum(), 2)
    return {"label": "Total Sales", "value": total_val}

# 2. KPI: Bulk Churn Stats for Dashboard
@app.get("/churn_stats")
def get_churn_stats():
    rfm = pd.read_csv('data/rfm_summary.csv')
    preds = churn_model.predict(rfm[['frequency', 'monetary']])
    at_risk = int(sum(preds))
    safe = len(preds) - at_risk
    return {
        "At Risk": at_risk,
        "Safe": safe,
        "Churn Rate": f"{round((at_risk/len(preds))*100, 2)}%"
    }

# 3. Graph: Sales Forecast
@app.get("/sales_forecast")
def get_forecast(days: int = 30):  
    future = ts_model.make_future_dataframe(periods=days) 
    forecast = ts_model.predict(future)
    res = forecast[['ds', 'yhat']].tail(days).to_dict('records')
    return {"forecast": res}
