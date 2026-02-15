import pandas as pd
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    data=pd.read_csv(file_path)
    return data
