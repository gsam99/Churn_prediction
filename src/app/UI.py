import gradio as gr
import requests
import os


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

def predict(id, channel_sales, cons_12m, cons_gas_12m, cons_last_month, date_activ,date_end,
            date_modif_prod,date_renewal,forecast_cons_12m,forecast_cons_year,forecast_discount_energy,forecast_meter_rent_12m,
            forecast_price_energy_off_peak,forecast_price_energy_peak,forecast_price_pow_off_peak,has_gas,imp_cons,margin_gross_pow_ele,
            margin_net_pow_ele,nb_prod_act,net_margin,num_years_antig,origin_up,pow_max,price_off_peak_var,
            price_peak_var,price_mid_peak_var,price_off_peak_fix,price_peak_fix,price_mid_peak_fix ):
    payload = {
        "client": {
            "id": id,
            "channel_sales": channel_sales,
            "cons_12m": cons_12m,
            "cons_gas_12m": cons_gas_12m,
            "cons_last_month": cons_last_month,
            "date_activ": "2020-01-01",
            "date_end": "2025-01-01",
            "date_modif_prod": "2021-01-01",
            "date_renewal": "2024-01-01",
            "forecast_cons_12m": forecast_cons_12m,
            "forecast_cons_year": forecast_cons_year,
            "forecast_discount_energy": forecast_discount_energy,
            "forecast_meter_rent_12m": forecast_meter_rent_12m,
            "forecast_price_energy_off_peak": forecast_price_energy_off_peak,
            "forecast_price_energy_peak": forecast_price_energy_peak,
            "forecast_price_pow_off_peak": forecast_price_pow_off_peak,
            "has_gas": False,
            "imp_cons": imp_cons,
            "margin_gross_pow_ele": margin_gross_pow_ele,
            "margin_net_pow_ele": margin_net_pow_ele,
            "nb_prod_act": nb_prod_act,
            "net_margin": net_margin,
            "num_years_antig": num_years_antig,
            "origin_up": origin_up,
            "pow_max": pow_max
        },
        "price": {
            "id": id,
            "price_date": "2024-01-01",
            "price_off_peak_var": price_off_peak_var,
            "price_peak_var": price_peak_var,
            "price_mid_peak_var": price_mid_peak_var,
            "price_off_peak_fix": price_off_peak_fix,
            "price_peak_fix": price_peak_fix,
            "price_mid_peak_fix": price_mid_peak_fix
        }
    }
    response = requests.post(API_URL, json=payload)
    print(response.json())
    if response.status_code ==200:
        print(response.json())
        result=response.json()
        w="Churned"
        if result["prediction"]==0:
            w="Not churned"
        else:
            w="Churned"

        return w
    else:
         return "Error in API call"

interface=gr.Interface(fn=predict,
                       inputs=[gr.Textbox(label="id"),
    gr.Textbox(label="channel_sales"),
    gr.Number(label="cons_12m"),
    gr.Number(label="cons_gas_12m"),
    gr.Number(label="cons_last_month"),
    gr.Textbox(label="date_activ", placeholder="YYYY-MM-DD"),
    gr.Textbox(label="date_end", placeholder="YYYY-MM-DD"),
    gr.Textbox(label="date_modif_prod", placeholder="YYYY-MM-DD"),
    gr.Textbox(label="date_renewal", placeholder="YYYY-MM-DD"),
    gr.Number(label="forecast_cons_12m"),
    gr.Number(label="forecast_cons_year"),
    gr.Number(label="forecast_discount_energy"),
    gr.Number(label="forecast_meter_rent_12m"),
    gr.Number(label="forecast_price_energy_off_peak"),
    gr.Number(label="forecast_price_energy_peak"),
    gr.Number(label="forecast_price_pow_off_peak"),
    gr.Checkbox(label="has_gas"),
    gr.Number(label="imp_cons"),
    gr.Number(label="margin_gross_pow_ele"),
    gr.Number(label="margin_net_pow_ele"),
    gr.Number(label="nb_prod_act"),
    gr.Number(label="net_margin"),
    gr.Number(label="num_years_antig"),
    gr.Textbox(label="origin_up"),
    gr.Number(label="pow_max"),
    gr.Number(label="price_off_peak_var"),
    gr.Number(label="price_peak_var"),
    gr.Number(label="price_mid_peak_var"),
    gr.Number(label="price_off_peak_fix"),
    gr.Number(label="price_peak_fix"),
    gr.Number(label="price_mid_peak_fix"),],
                       outputs=gr.Textbox(label="Prediction Result"),
                       title="Churn Prediction System")

interface.launch()

