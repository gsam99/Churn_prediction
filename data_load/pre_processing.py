import pandas as pd
import numpy as np

def aggregate_fun(data,col,agg_dict):
    df=(data.groupby(col).agg(agg_dict).reset_index())
    return df
def jan_dec_diff(data):
    jan_data=data.groupby("id").first().reset_index()
    dec_data=data.groupby("id").last().reset_index()
    diff_data=pd.merge(jan_data.rename(columns={"price_off_peak_var":"jan_var","price_off_peak_fix":"jan_fix"}),dec_data,on="id")
    diff_data["off_peak_var_jan_dec"]=diff_data["jan_var"]-diff_data["price_off_peak_var"]
    diff_data["off_peak_fix_jan_dec"]=diff_data["jan_fix"]-diff_data["price_off_peak_fix"]
    diff_data.drop(columns=["price_off_peak_var","price_off_peak_fix","jan_var","jan_fix","price_date_y","price_date_x"],inplace=True)
    return diff_data
def cycles(data,col1,col2,desc="mean_by_month"):
    new_col=desc+col1+"_"+col2
    data[new_col]=data[col1]-data[col2]
    return data

def rare_samples_checking(data,col,min_samples=100):
    value_count_dict=data[col].value_counts().to_dict()
    cat_list=[]
    for v, i in value_count_dict.items():
        if i>=min_samples:
            cat_list.append(v)
    return cat_list


def join_tables(df1,df2,merge_col):
    df=pd.merge(df1,df2,on=merge_col)
    return df


def data_pre_processing(client_data,price_data):
    client_date_columns=["date_activ","date_end","date_modif_prod","date_renewal"]
    for d in client_date_columns:
        client_data[d]=pd.to_datetime(client_data[d],format='%Y-%m-%d')
    client_data["has_gas"]=client_data["has_gas"].map({"t":1,"f":0})
    price_data["price_date"]=pd.to_datetime(price_data["price_date"],format='%Y-%m-%d')
    monthly_price=aggregate_fun(price_data,["id","price_date"],{
        "price_off_peak_var": "mean",
        "price_off_peak_fix": "mean"
    })
    diff_data=jan_dec_diff(monthly_price)
    final_data=join_tables(diff_data,client_data,"id")
    price_avgs=aggregate_fun(price_data,["id"],{"price_off_peak_var":"mean","price_peak_var":"mean","price_mid_peak_var":"mean","price_off_peak_fix":"mean","price_peak_fix":"mean","price_mid_peak_fix":"mean"})
    price_avgs=cycles(price_avgs,"price_off_peak_var","price_peak_var","mean")
    price_avgs=cycles(price_avgs,"price_peak_var","price_mid_peak_var","mean")
    price_avgs=cycles(price_avgs,"price_off_peak_var","price_mid_peak_var","mean")
    price_avgs=cycles(price_avgs,"price_off_peak_fix","price_peak_fix","mean")
    price_avgs=cycles(price_avgs,"price_peak_fix","price_mid_peak_fix","mean")
    price_avgs=cycles(price_avgs,"price_off_peak_fix","price_mid_peak_fix","mean")
    final_data=join_tables(final_data,price_avgs,"id")
    prices_agg_monthly=aggregate_fun(price_data,["id","price_date"],{"price_off_peak_var":"mean","price_peak_var":"mean","price_mid_peak_var":"mean","price_off_peak_fix":"mean","price_peak_fix":"mean","price_mid_peak_fix":"mean"})
    prices_agg_monthly=cycles(prices_agg_monthly,"price_off_peak_var","price_peak_var","monthly_mean")
    prices_agg_monthly=cycles(prices_agg_monthly,"price_peak_var","price_mid_peak_var","monthly_mean")
    prices_agg_monthly=cycles(prices_agg_monthly,"price_off_peak_var","price_mid_peak_var","monthly_mean")
    prices_agg_monthly=cycles(prices_agg_monthly,"price_off_peak_fix","price_peak_fix","monthly_mean")
    prices_agg_monthly=cycles(prices_agg_monthly,"price_off_peak_fix","price_peak_fix","monthly_mean")
    prices_agg_monthly=cycles(prices_agg_monthly,"price_off_peak_fix","price_mid_peak_fix","monthly_mean")
    prices_agg_monthly.drop(columns=["price_date","price_off_peak_var","price_peak_var","price_mid_peak_var","price_off_peak_fix","price_peak_fix","price_mid_peak_fix"],inplace=True)
    final_data=join_tables(final_data,prices_agg_monthly,"id")
    final_data["Tenure"]=((final_data["date_end"]-final_data["date_activ"]).dt.days//365)

    dummies = pd.get_dummies(final_data["channel_sales"], dtype=int)
    rare_cats = rare_samples_checking(final_data, "channel_sales")
    dummies = dummies[rare_cats]
    final_data = pd.concat([final_data, dummies], axis=1)

    dummies_origin = pd.get_dummies(final_data["origin_up"], prefix="origin_up", dtype=int)
    rare_cats_origin = rare_samples_checking(final_data, "origin_up")
    keep_cols = [f"origin_up_{c}" for c in rare_cats_origin]
    dummies_origin = dummies_origin[keep_cols]
    final_data=pd.concat([final_data,dummies_origin],axis=1)

    final_data=final_data.drop(columns=["channel_sales","origin_up"])

    print(final_data.info())





    return final_data


file_client_data="D:\\Projects\\Churn_prediction_model\\Data\\client_data.csv"
file_price_data="D:\\Projects\\Churn_prediction_model\\Data\\price_data.csv"
client_data=pd.read_csv(file_client_data)
price_data=pd.read_csv(file_price_data)
difference_data=data_pre_processing(client_data,price_data)
print(difference_data)

