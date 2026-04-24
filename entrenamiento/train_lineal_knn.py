import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def entrenar_lineal_knn(test_size=0.2, k=5):
    df = pd.read_csv("dataset_ciclismo_fatiga.csv")

    X = df.drop("fatiga", axis=1)
    y = df["fatiga"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Guardar test
    joblib.dump((X_test, y_test), "test_set.pkl")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(lr, "modelo_lr.pkl")
    joblib.dump(knn, "modelo_knn.pkl")

    return "Modelo entrenado correctamente"


def entrenar_modelo(test_size=0.2, k=5):
    # Alias para compatibilidad con código previo.
    return entrenar_lineal_knn(test_size=test_size, k=k)