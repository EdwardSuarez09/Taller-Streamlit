import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def entrenar_arbol(test_size=0.2):
    df = pd.read_csv("dataset_ciclismo_fatiga.csv")

    X = df.drop("fatiga", axis=1)
    y = df["fatiga"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    modelo = DecisionTreeRegressor()
    modelo.fit(X_train, y_train)

    joblib.dump(modelo, "modelo_arbol.pkl")

    return "Árbol entrenado"