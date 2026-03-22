from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date
import pandas as pd
import sys
import os

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from data_load.data_loader import load_data
from data_load.pre_processing import data_pre_processing
import joblib
import gradio as gr

client_path="D:\\Projects\\Churn_prediction_model\\Data\\client_data.csv"
price_path="D:\\Projects\\Churn_prediction_model\\Data\\price_data.csv"

client_data =load_data(client_path)
price_data=load_data(price_path)

model=joblib.load("D:\\Projects\\Churn_prediction_model\\artifacts\\model.pkl")
import json

features_path = "D:\\Projects\\Churn_prediction_model\\artifacts\\X_features.json"

with open(features_path, "r") as f:
    feature_columns = json.load(f)

#print(feature_columns)
app=FastAPI(title="Churn Prediction System",
            description="ML predicting customers for energy industry",
            version="1.0.0")

@app.get("/")
def root():
    return {"status":"ok"}

class ClientData(BaseModel):
    id: str
    channel_sales: str
    cons_12m: float
    cons_gas_12m: float
    cons_last_month: float
    date_activ: date
    date_end: date
    date_modif_prod:date
    date_renewal:date
    forecast_cons_12m: float
    forecast_cons_year: float
    forecast_discount_energy: float
    forecast_meter_rent_12m: float
    forecast_price_energy_off_peak:float
    forecast_price_energy_peak:float
    forecast_price_pow_off_peak:float
    has_gas:bool
    imp_cons:float
    margin_gross_pow_ele:float
    margin_net_pow_ele: float
    nb_prod_act: float
    net_margin: float
    num_years_antig:float
    origin_up: str
    pow_max: float

class PriceData(BaseModel):
    id: str
    price_date: date
    price_off_peak_var: float
    price_peak_var: float
    price_mid_peak_var: float
    price_off_peak_fix: float
    price_peak_fix: float
    price_mid_peak_fix: float

@app.post("/predict")
def predict(client: ClientData,price: PriceData):
    client_df=pd.DataFrame([client.dict()])
    price_df=pd.DataFrame([price.dict()])
    processed_data=data_pre_processing(client_df,price_df)
    processed_df = processed_data.reindex(columns=feature_columns, fill_value=0)
    prediction=model.predict(processed_df)
    return {"prediction":int(prediction[0])}





