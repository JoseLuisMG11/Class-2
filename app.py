import streamlit as st
from text_classification import run_text_app
from image_classification import run_image_app
from regression_model import run_regression_app
from about import show_about

st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Ir a", ["Clasificación de Texto", "Clasificación de Imagen", "Regresión", "Acerca de"])

if page == "Clasificación de Texto":
    run_text_app()
elif page == "Clasificación de Imagen":
    run_image_app()
elif page == "Regresión":
    run_regression_app()
else:
    show_about()