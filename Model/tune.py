import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def tune_model(X,Y):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": 42,
        }
        model=RandomForestClassifier(**params)
        scores=cross_val_score(model,X,Y,cv=3,scoring="recall")
        return scores.mean()
    study=optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=20)
    print("Best Parameters:",study.best_params)
    return study.best_params