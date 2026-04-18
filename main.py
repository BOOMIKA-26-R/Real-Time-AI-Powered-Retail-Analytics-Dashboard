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

@app.get("/total_sales")
def get_total_sales(simulation_multiplier: float = 1.0):
    # Load real data
    file_path = os.path.join(BASE_DIR, 'data', 'olist_order_items_dataset.csv')
    if not os.path.exists(file_path):
        return {"error": "Data file not found"}
    
    items_df = pd.read_csv(file_path)
    
    # Calculate real revenue
    real_revenue = items_df['price'].sum()
    
    # SIMULATION LOGIC: Multiply by the parameter (Default is 1.0, so no change)
    simulated_revenue = real_revenue * simulation_multiplier
    
    return {
        "label": "Total Sales",
        "value": round(simulated_revenue, 2),
        "note": "Simulation Active" if simulation_multiplier != 1.0 else "Real Data"
    }

@app.get("/churn_stats")
def get_churn_stats(risk_threshold: float = 0.5):
    file_path = os.path.join(BASE_DIR, 'data', 'rfm_summary.csv')
    if not os.path.exists(file_path):
        return {"error": "Data file not found"}

    rfm = pd.read_csv(file_path)
    
    # Get probability of churn (0 to 1)
    probs = churn_model.predict_proba(rfm[['frequency', 'monetary']])[:, 1]
    
    
    at_risk_count = int((probs > risk_threshold).sum())
    safe_count = len(rfm) - at_risk_count
    
    return {
        "At Risk": at_risk_count,
        "Safe": safe_count,
        "Churn Rate": f"{round((at_risk_count / len(rfm)) * 100, 1)}%"
    }


# 3. Graph: Sales Forecast
@app.get("/sales_forecast")
def get_forecast(days: int = 30):  
    future = ts_model.make_future_dataframe(periods=days) 
    forecast = ts_model.predict(future)
    res = forecast[['ds', 'yhat']].tail(days).to_dict('records')
    return {"forecast": res}
