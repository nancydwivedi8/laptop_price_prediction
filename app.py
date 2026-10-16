import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻"
)

st.title("Laptop Price Prediction")

MODEL_PATH = Path(__file__).parent / "laptop_price_prediction.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.write("Your laptop prediction model is ready.")
