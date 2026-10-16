import streamlit as st
import pandas as pd
import joblib
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "Company": company,
        "Ram": ram,
        "Weight": weight,
        "CPU": cpu,
        "GPU": gpu,
        "HDD": hdd,
        "SSD": ssd
    }])

    # Model ke training columns se match karo
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
