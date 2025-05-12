import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_and_preprocess_data():
    config = load_config()

    # Cargar datos
    df = pd.read_csv(config["data_path"])

    # Separar features y target
    X = df.drop("class", axis=1)
    y = df["class"]

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"]
    )

    return X_train, X_test, y_train, y_test