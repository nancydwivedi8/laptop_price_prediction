import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("laptop_price_prediction.pkl")

st.title("Laptop Price Prediction")

# User inputs
company = st.selectbox(
    "Company",
    ["Apple", "Asus", "Dell", "HP", "Lenovo", "Acer", "MSI"]
)

typename = st.selectbox(
    "Laptop Type",
    ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible", "Workstation"]
)

ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5)
cpu = st.selectbox(
    "CPU",
    ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD Processor"]
)
gpu = st.selectbox("GPU", ["Intel", "Nvidia", "AMD"])
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 1000, 2000])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1000, 2000])
ips = st.selectbox("IPS Display", [0, 1])
screen_size = st.number_input(
    "Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6
)
resolution = st.selectbox(
    "Screen Resolution",
    ["1920x1080", "1366x768", "1600x900", "3840x2160"]
)
touchscreen = st.selectbox("Touchscreen", [0, 1])
os_name = st.selectbox(
    "Operating System",
    ["Windows", "Mac", "Linux", "Other"]
)

if st.button("Predict Price"):

    input_data = pd.DataFrame([{
        "Company": company,
        "TypeName": typename,
        "Ram": ram,
        "Weight": weight,
        "CPU": cpu,
        "GPU": gpu,
        "HDD": hdd,
        "SSD": ssd,
        "Ips": ips,
        "Ppi": screen_size,
        "Touchscreen": touchscreen,
        "OpSys": os_name,
        "ScreenResolution": resolution
    }])

    # Match the model's expected feature names
    expected_columns = model.feature_names_in_

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Laptop Price: ₹{prediction[0]:,.2f}")
