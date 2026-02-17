# create complete pipeline-> data loading -> feature building -> model building ->model evaluation

import pandas as pd
import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_load.data_loader import load_data
from data_load.pre_processing import data_pre_processing


def main(args):
    if len(args)<1:
        print("Please provide arguments")
        sys.exit(1)
    client_filename= "D:\\Projects\\Churn_prediction_model\\Data\\"+args[0]
    price_filename="D:\\Projects\\Churn_prediction_model\\Data\\"+args[1]
    client_data=load_data(client_filename)
    price_data=load_data(price_filename)
    ml_run_path=args.mlflow.uri or f"file://{project_root}/mlruns"



if __name__ == "__main__":
    main(sys.argv[1:])

