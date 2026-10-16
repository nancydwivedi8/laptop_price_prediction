
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Laptop Price Prediction", page_icon="💻")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("laptop_price_prediction.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

st.title("💻 Laptop Price Prediction")
st.write("Laptop ki details select karke price predict karein.")

expected_columns = list(model.feature_names_in_)

# Original categorical features used in laptop datasets
categorical_features = [
    "Company", "TypeName", "ScreenResolution",
    "CPU", "GPU", "OpSys", "Memory", "Ram"
]

input_values = {}

with st.form("laptop_form"):
    st.subheader("Laptop Details")

    # Build dropdowns from categories learned during training
    for feature in categorical_features:
        prefix = feature + "_"
        options = [
            col[len(prefix):]
            for col in expected_columns
            if col.startswith(prefix)
        ]

        if options:
            input_values[feature] = st.selectbox(
                feature, sorted(set(options))
            )

    # Numeric columns that exist in the trained model
    numeric_defaults = {
        "Inches": (15.6, 10.0, 20.0),
        "Ram": (8.0, 1.0, 128.0),
        "Weight": (1.5, 0.3, 10.0),
        "HDD": (0.0, 0.0, 4000.0),
        "SSD": (256.0, 0.0, 4000.0),
        "Ips": (1.0, 0.0, 1.0),
        "Touchscreen": (0.0, 0.0, 1.0),
        "Ppi": (141.0, 30.0, 500.0)
    }

    for feature, settings in numeric_defaults.items():
        # Only show numeric inputs if the model expects this column
        if feature in expected_columns:
            default, minimum, maximum = settings
            input_values[feature] = st.number_input(
                feature,
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default)
            )

    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        # Convert user inputs to model's expected feature format
        input_df = pd.DataFrame([input_values])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(
            columns=expected_columns,
            fill_value=0
        )

        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Laptop Price: ₹{prediction:,.2f}")

        with st.expander("View model input"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction error: {e}")
