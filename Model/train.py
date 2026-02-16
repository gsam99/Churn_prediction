import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df,target_column):
    df=df.drop("id",axis=1)
    X=df.drop(columns=[target_column])
    Y=df[target_column]

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    rf=RandomForestClassifier(n_estimators=100,max_features=5,n_jobs=-1,random_state=42)


    with mlflow.start_run():
        rf.fit(x_train, y_train)
        y_pred=rf.predict(x_test)
        recall=recall_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        mlflow.log_param("n_estimators",rf.n_estimators)
        mlflow.log_param("max_features",rf.max_features)
        mlflow.log_param("n_jobs",rf.n_jobs)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("precision",precision)
        mlflow.sklearn.log_model(rf,"random Forest Model")
        train_df=mlflow.data.from_pandas(df,source="training_data")
        mlflow.log_input(train_df,context="Training Data")
        print(f'recall:{recall}, precision:{precision}')

file_name="D:\Projects\Churn_prediction_model\Data\data_for_predictions.csv"
df=pd.read_csv(file_name)
target_column="churn"
train_model(df,target_column)

