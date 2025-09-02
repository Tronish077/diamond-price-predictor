import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model only
model = joblib.load(open("diamond_price_model.pkl", "rb"))

st.title("💎 Diamond Price Predictor")

# Inputs (matching your makePrediction function)
carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.number_input("Depth", min_value=40.0, max_value=80.0, value=61.0, step=0.1)
table = st.number_input("Table", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
x = st.number_input("X (mm)", min_value=0.0, max_value=15.0, value=5.0, step=0.01)
y = st.number_input("Y (mm)", min_value=0.0, max_value=15.0, value=5.0, step=0.01)
z = st.number_input("Z (mm)", min_value=0.0, max_value=15.0, value=3.0, step=0.01)

if st.button("Predict Price"):
    # Wrap inputs in DataFrame
    df = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    # Directly predict (model must include preprocessing inside the pipeline)
    y_pred_log = model.predict(df)
    y_pred = np.exp(y_pred_log)

    st.success(f"💰 Predicted Diamond Price: $ {y_pred[0]:,.2f}")
    st.write("Input DF:", df)
    st.write("Model expects:", model.feature_names_in_)
