import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------
# Title
# -----------------------
st.title("🚗 Car Price Predictor (KNN Model)")

# -----------------------
# Encode Car Brands
# -----------------------
model_encoding = {
    "Toyota": 0,
    "Honda": 1,
    "Ford": 2
}

# -----------------------
# Sample Training Data
# -----------------------
X = np.array([
    [0, 2015, 50000, 1500, 0, 1],
    [1, 2017, 30000, 1600, 0, 1],
    [2, 2012, 80000, 1400, 1, 0],
    [0, 2020, 10000, 2000, 0, 1],
    [1, 2018, 40000, 1800, 1, 1]
])

y = np.array([
    15000,
    18000,
    9000,
    25000,
    20000
])

# -----------------------
# Feature Scaling
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# Train Model
# -----------------------
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_scaled, y)

# -----------------------
# User Inputs
# -----------------------
st.header("Enter Car Details")

brand = st.selectbox("Car Brand", list(model_encoding.keys()))

year = st.number_input(
    "Model Year",
    min_value=1990,
    max_value=2026,
    value=2018
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    max_value=300000,
    value=50000
)

engine = st.number_input(
    "Engine Size (cc)",
    min_value=500,
    max_value=5000,
    value=1500
)

accident = st.selectbox("Accident History", ["No", "Yes"])

clean_title = st.selectbox("Clean Title", ["No", "Yes"])

# -----------------------
# Convert Inputs
# -----------------------
brand_encoded = model_encoding[brand]
accident_value = 1 if accident == "Yes" else 0
title_value = 1 if clean_title == "Yes" else 0

# -----------------------
# Prepare Input
# -----------------------
input_data = np.array([
    [brand_encoded, year, mileage, engine, accident_value, title_value]
])

input_scaled = scaler.transform(input_data)

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Price"):

    price = model.predict(input_scaled)

    st.success(f"💰 Predicted Car Price: ${price[0]:,.2f}")

# -----------------------
# Optional: Show Dataset
# -----------------------
if st.checkbox("Show Training Data"):
    st.write("Features:", X)
    st.write("Prices:", y)
