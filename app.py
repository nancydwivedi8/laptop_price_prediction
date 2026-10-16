import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Laptop Price Prediction")
st.write("Enter laptop details to estimate its price.")

# Load model
MODEL_PATH = Path(__file__).parent / "laptop_price_prediction.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.success("Model loaded successfully!")

# Laptop details
company = st.selectbox(
    "Company",
    ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Other"]
)

typename = st.selectbox(
    "Laptop Type",
    ["Notebook", "Ultrabook", "Gaming",
     "2 in 1 Convertible", "Workstation", "Other"]
)

ram = st.selectbox("RAM (GB)", [4, 8, 16, 24, 32, 64])
weight = st.number_input("Weight (kg)", 0.5, 6.0, 1.8)
screen_size = st.number_input(
    "Screen Size (inches)", 10.0, 20.0, 15.6
)

touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])

resolution = st.selectbox(
    "Screen Resolution",
    ["1920x1080", "1366x768", "2560x1440",
     "3840x2160", "1600x900", "Other"]
)

cpu = st.selectbox(
    "CPU",
    ["Intel Core i3", "Intel Core i5", "Intel Core i7",
     "Intel Core i9", "AMD Ryzen 3", "AMD Ryzen 5",
     "AMD Ryzen 7", "Other"]
)

hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 1000, 2000])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1000, 2000])
gpu = st.selectbox("GPU", ["Intel", "Nvidia", "AMD", "Apple", "Other"])
opsys = st.selectbox(
    "Operating System",
    ["Windows", "Mac", "Linux", "Chrome OS", "Other"]
)

# Prediction button
if st.button("Predict Price", type="primary"):
    try:
        input_data = pd.DataFrame([{
            "Company": company,
            "TypeName": typename,
            "Ram": ram,
            "Weight": weight,
            "Touchscreen": 1 if touchscreen == "Yes" else 0,
            "Ips": 1 if ips == "Yes" else 0,
            "ScreenSize": screen_size,
            "ScreenResolution": resolution,
            "Cpu": cpu,
            "HDD": hdd,
            "SSD": ssd,
            "Gpu": gpu,
            "OpSys": opsys
        }])

        prediction = model.predict(input_data)
        price = float(prediction[0])

        st.success(f"Predicted Laptop Price: ₹{price:,.2f}")

    except Exception as e:
        st.error("Prediction failed. Model ke input features check karein.")
        st.exception(e)
