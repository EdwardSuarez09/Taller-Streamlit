import joblib
from sklearn.metrics import mean_squared_error, r2_score

def evaluar_modelos():
    X_test, y_test = joblib.load("test_set.pkl")
    scaler = joblib.load("scaler.pkl")

    lr = joblib.load("modelo_lr.pkl")
    knn = joblib.load("modelo_knn.pkl")
    arbol = joblib.load("modelo_arbol.pkl")

    X_test_scaled = scaler.transform(X_test)

    y_lr = lr.predict(X_test_scaled)
    y_knn = knn.predict(X_test_scaled)
    y_arbol = arbol.predict(X_test)

    return {
        "LR": {
            "MSE": mean_squared_error(y_test, y_lr),
            "R2": r2_score(y_test, y_lr)
        },
        "KNN": {
            "MSE": mean_squared_error(y_test, y_knn),
            "R2": r2_score(y_test, y_knn)
        },
        "ARBOL": {
            "MSE": mean_squared_error(y_test, y_arbol),
            "R2": r2_score(y_test, y_arbol)
        }
    }