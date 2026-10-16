import streamlit as st
import joblib
import pandas as pd
import sklearn

# Load trained model
model = joblib.load("laptop_price_prediction.pkl")

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Laptop Price Prediction")
st.write("Enter laptop specifications to predict its price.")

st.divider()

# User inputs
company = st.selectbox(
    "Laptop Company",
    ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Other"]
)

typename = st.selectbox(
    "Laptop Type",
    ["Notebook", "Gaming", "Ultrabook", "2 in 1 Convertible", "Workstation"]
)

ram = st.selectbox("RAM (GB)", [2, 4, 8, 12, 16, 24, 32, 64])

weight = st.number_input(
    "Weight (kg)", min_value=0.5, max_value=10.0,
    value=2.0, step=0.1
)

touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])

screen_size = st.number_input(
    "Screen Size (inches)", min_value=10.0,
    max_value=25.0, value=15.6, step=0.1
)

resolution = st.selectbox(
    "Screen Resolution",
    ["1366x768", "1600x900", "1920x1080",
     "2560x1440", "3840x2160"]
)

cpu = st.selectbox(
    "CPU Brand",
    ["Intel Core i3", "Intel Core i5", "Intel Core i7",
     "Intel Core i9", "AMD", "Other"]
)

hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 1000, 2000])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1000, 2000])

gpu = st.selectbox(
    "GPU Brand", ["Intel", "Nvidia", "AMD", "Other"]
)

os_name = st.selectbox(
    "Operating System", ["Windows", "Mac", "Linux", "Other"]
)

if st.button("Predict Laptop Price", type="primary"):
    st.warning(
        "Make sure the input columns and their order match "
        "the features used to train your model."
    )

    # This example assumes the model was trained using
    # these exact feature names and raw categorical values.
    input_data = pd.DataFrame([{
        "Company": company,
        "TypeName": typename,
        "Ram": ram,
        "Weight": weight,
        "Touchscreen": 1 if touchscreen == "Yes" else 0,
        "Ips": 1 if ips == "Yes" else 0,
        "ScreenSize": screen_size,
        "Resolution": resolution,
        "CPU": cpu,
        "HDD": hdd,
        "SSD": ssd,
        "GPU": gpu,
        "OpSys": os_name
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success("Predicted Laptop Price")
        st.subheader(f"₹ {prediction:,.2f}")
    except Exception as e:
        st.error(
            "Prediction failed because the input features "
            "do not match the trained model."
        )
        st.code(str(e))

st.caption(f"Scikit-learn version: {sklearn.__version__}")
