# create complete pipeline-> data loading -> feature building -> model building ->model evaluation
import argparse
import pandas as pd
import numpy as np
import mlflow
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_load.data_loader import load_data
from data_load.pre_processing import data_pre_processing
from Model.train import train_model
from Model.tune import tune_model
from Model.evaluate import evaluate_model


def main( ):
    #---------------------Args Parser----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_file", required=True)
    parser.add_argument("--price_file", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--experiment", default="Churn_prediction")
    parser.add_argument("--mlflow_uri", default=None)
    args = parser.parse_args()
    #---------------------ML configuration------------------------
    ml_run_path = args.mlflow_uri or f"sqlite:///{project_root}/mlflow.db"
    mlflow.set_tracking_uri(ml_run_path)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_param("model","Random Forest Classifier")
        mlflow.log_param("threshold",args.threshold)
        mlflow.log_param("test_size",args.test_size)
    #-------------------data loading---------------------------
    data_dir = os.path.join(project_root, "Data")
    client_filename = os.path.join(data_dir, args.client_file)
    price_filename = os.path.join(data_dir, args.price_file)
    client_data = load_data(client_filename)
    price_data = load_data(price_filename)

    #---------------------------pre-processing--------------------
    df=data_pre_processing(client_data,price_data)
    target=df["churn"]

    #----------------------------Meta Data consistency-------------
    import json,joblib
    artifacts_dir=os.path.join(project_root,"artifacts")
    os.makedirs(artifacts_dir,exist_ok=True)

    X_features=list(df.drop(columns=["churn"]).columns)
    with open(os.path.join(artifacts_dir,"X_features.json"),"w") as f:
        json.dump(X_features,f)
    mlflow.log_text("\n".join(X_features),artifact_file="X_features.txt")
    preprocessing_artifact={"features columns":X_features,
                            "target columns":target}
    joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
    mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))

    #-------------------------------Train/Test Split-----------------
    X=df.drop(columns=["churn"])
    Y=df["churn"]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=args.test_size,random_state=42)

    #-------------------------------Model Tuning----------------------
    param=None
    best_params=tune_model(x_train,y_train)
    params={**best_params,"n_jobs":-1,"random_state":42}
    for k, v in params.items():
        mlflow.log_param(k,v)

    #-------------------------------Train Model------------------------
    model=train_model(x_train,x_test,y_train,y_test,args.threshold)

    #-------------------------------Evaluation----------------------
    evaluate_model(model,x_test,y_test)
    model_path = os.path.join(artifacts_dir, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)








if __name__ == "__main__":
    main()

