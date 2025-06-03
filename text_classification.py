# text_classification.py
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

def run_text_app():
    st.title("Clasificación de Texto con Deep Learning")

    # Cargar modelo y tokenizer
    model = load_model("models/text_model.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    max_length = 100

    # Entrada de texto
    user_input = st.text_area("Introduce un texto para clasificar")

    if st.button("Clasificar"):
        if user_input.strip() == "":
            st.warning("Por favor escribe algo.")
        else:
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=max_length, padding='post')
            prediction = model.predict(padded)[0][0]
            label = "Positivo" if prediction > 0.5 else "Negativo"

            st.write(f"**Resultado:** {label} ({prediction:.2f})")
