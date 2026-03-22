import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

def train_model(x_train, x_test, y_train, y_test, threshold=0.5, params=None):
    params = params or {"n_estimators": 50, "random_state": 42, "n_jobs": -1}
    model = RandomForestClassifier(**params)

    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))

    return model