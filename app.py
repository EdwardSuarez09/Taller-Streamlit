import streamlit as st
from entrenamiento.train_lineal_knn import entrenar_lineal_knn
from entrenamiento.train_arbol import entrenar_arbol
from test.test_modelos import evaluar_modelos
import os

st.set_page_config(layout="wide")
st.title("Predicción de Fatiga - Modelos ML")

st.sidebar.header("Configuración")
test_size = st.sidebar.selectbox("Test size", [0.2, 0.3])
k_val = st.sidebar.slider("K vecinos (KNN)", 1, 20, 5)

# 🔹 ENTRENAR
if st.sidebar.button("Entrenar modelos"):
    with st.spinner("Entrenando..."):
        msg1 = entrenar_lineal_knn(test_size, k_val)
        msg2 = entrenar_arbol(test_size)

        st.success(msg1)
        st.success(msg2)

# 🔹 EVALUAR
if st.sidebar.button("Evaluar modelos"):
    if not os.path.exists("modelo_lr.pkl"):
        st.error("Primero entrena los modelos")
    else:
        metrics = evaluar_modelos()

        col1, col2, col3 = st.columns(3)

        col1.subheader("Regresión Lineal")
        col1.metric("MSE", f"{metrics['LR']['MSE']:.2f}")
        col1.metric("R2", f"{metrics['LR']['R2']:.3f}")

        col2.subheader("KNN")
        col2.metric("MSE", f"{metrics['KNN']['MSE']:.2f}")
        col2.metric("R2", f"{metrics['KNN']['R2']:.3f}")

        col3.subheader("Árbol")
        col3.metric("MSE", f"{metrics['ARBOL']['MSE']:.2f}")
        col3.metric("R2", f"{metrics['ARBOL']['R2']:.3f}")