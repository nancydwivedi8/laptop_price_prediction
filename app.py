import streamlit as st
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "laptop_price_prediction.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()
