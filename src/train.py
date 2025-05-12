import os
import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from preprocess import load_and_preprocess_data
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model(X_train, X_test, y_train, y_test):
    config = load_config()

    # Configuración de XGBoost
    model = xgb.XGBClassifier(
        n_estimators=config["model"]["params"]["n_estimators"],
        learning_rate=config["model"]["params"]["learning_rate"],
        max_depth=config["model"]["params"]["max_depth"]
    )

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, accuracy, mse, r2

def main():
    # Cargar datos preprocesados
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Iniciar el tracking de MLflow
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    mlflow.start_run()

    # Registrar parámetros
    mlflow.log_param("n_estimators", config["model"]["params"]["n_estimators"])
    mlflow.log_param("learning_rate", config["model"]["params"]["learning_rate"])
    mlflow.log_param("max_depth", config["model"]["params"]["max_depth"])

    # Entrenar y evaluar el modelo
    model, accuracy, mse, r2 = train_model(X_train, X_test, y_train, y_test)

    # Registrar métricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Registrar el modelo
    mlflow.xgboost.log_model(model, "xgboost_model")

    # Finalizar el run
    mlflow.end_run()

    # Mostrar resultados
    print(f"Accuracy: {accuracy}")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")

if __name__ == "__main__":
    main()
