import pandas as pd
import joblib
from xgboost import XGBClassifier
from prophet import Prophet

# Load Data
orders = pd.read_csv('data/olist_orders_dataset.csv')
customers = pd.read_csv('data/olist_customers_dataset.csv')
items = pd.read_csv('data/olist_order_items_dataset.csv')

# Preprocessing
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
master_df = orders.merge(customers, on='customer_id').merge(items, on='order_id')
snapshot_date = master_df['order_purchase_timestamp'].max()

# Create RFM Table
rfm = master_df.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'price': 'sum'
}).rename(columns={'order_purchase_timestamp': 'recency', 'order_id': 'frequency', 'price': 'monetary'})

# Label Churn (1 if no purchase in 90 days)
rfm['churn'] = (rfm['recency'] > 90).astype(int)

# 1. Train & Save Churn Model
churn_model = XGBClassifier().fit(rfm[['frequency', 'monetary']], rfm['churn'])
joblib.dump(churn_model, 'churn_model.pkl')
# Save RFM for the API to use in bulk stats
rfm.to_csv('data/rfm_summary.csv', index=False)

# 2. Train & Save Sales Forecast (Prophet)
forecast_df = master_df.resample('D', on='order_purchase_timestamp')['price'].sum().reset_index()
forecast_df.columns = ['ds', 'y']
ts_model = Prophet().fit(forecast_df)
joblib.dump(ts_model, 'sales_forecast.pkl')

print("All models and data summaries saved successfully!")
import pandas as pd
import joblib
from xgboost import XGBClassifier
from prophet import Prophet

# Load Data
orders = pd.read_csv('data/olist_orders_dataset.csv')
customers = pd.read_csv('data/olist_customers_dataset.csv')
items = pd.read_csv('data/olist_order_items_dataset.csv')

# Preprocessing
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
master_df = orders.merge(customers, on='customer_id').merge(items, on='order_id')
snapshot_date = master_df['order_purchase_timestamp'].max()

# Create RFM Table
rfm = master_df.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'price': 'sum'
}).rename(columns={'order_purchase_timestamp': 'recency', 'order_id': 'frequency', 'price': 'monetary'})

# Label Churn (1 if no purchase in 90 days)
rfm['churn'] = (rfm['recency'] > 90).astype(int)

# 1. Train & Save Churn Model
churn_model = XGBClassifier().fit(rfm[['frequency', 'monetary']], rfm['churn'])
joblib.dump(churn_model, 'churn_model.pkl')
# Save RFM for the API to use in bulk stats
rfm.to_csv('data/rfm_summary.csv', index=False)

# 2. Train & Save Sales Forecast (Prophet)
forecast_df = master_df.resample('D', on='order_purchase_timestamp')['price'].sum().reset_index()
forecast_df.columns = ['ds', 'y']
ts_model = Prophet().fit(forecast_df)
joblib.dump(ts_model, 'sales_forecast.pkl')

print("All models and data summaries saved successfully!")
