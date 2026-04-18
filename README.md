**Project Overview**

This project is a complete production-ready system that combines Machine Learning, FastAPI, and Power BI to provide real-time retail insights. It identifies customer churn risks and forecasts future sales using live data streaming simulations.

**Key Objectives**

- Churn Prediction: Identifying at-risk customers using an XGBoost classifier.

- Sales Forecasting: Predicting 30-day revenue trends using Facebook Prophet.

- Live Dashboarding: Real-time data visualization via Power BI connected to a cloud API.

- Cloud Deployment: Fully hosted backend on Render.

**Tech Stack**

- Language: Python 3.10

- Framework: FastAPI (High-performance web API)

- Machine Learning: Scikit-learn, XGBoost, Prophet

- Data Handling: Pandas, Joblib

- Visualization: Power BI Desktop

- Cloud Hosting: Render (Web Services)

 - Version Control: Git & GitHub

**Dashboard Visuals**


<img width="1323" height="732" alt="2026-04-18" src="https://github.com/user-attachments/assets/39561924-774f-49b7-ab5c-dfaabb15c3ac" />






**How It Works**

- Model Training (train.py): Processes the Brazilian E-Commerce dataset (Olist) to train predictive models and generates .pkl files.

- API Layer (main.py): A FastAPI server that exposes endpoints for Sales Forecasting, Churn Statistics, and Total Sales.

- Real-Time Connection: Power BI fetches data from the live Render API using JSON web requests. Every "Refresh" trigger recalculates the AI models.

**API Endpoints**

- GET /total_sales: Returns the live total gross revenue.

- GET /churn_stats: Returns the count of "Safe" vs "At Risk" customers.

- GET /sales_forecast: Returns a 30-day predicted sales trend.

- GET /docs: Interactive Swagger UI for API testing.
  
**Deployment Link**

Live API: https://real-time-ai-powered-retail-analytics.onrender.com

**Project Structure**


├── **data**/

│   ├── olist_order_items_dataset.csv


│   └── rfm_summary.csv

├── **main.py**                 # FastAPI Application

├── **train.py**                # ML Training Script

├── **requirements.txt**         # Library dependencies

├── **churn_model.pkl**         # Pre-trained XGBoost Model

├── **sales_forecast.pkl**    # Pre-trained Prophet Model

└── **Retail_Dashboard.pbix**  # Power BI Dashboard File
