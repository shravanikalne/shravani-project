import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Title
# -----------------------------
st.title("🚗 Car Price Predictor (KNN Model)")

# -----------------------------
# Model Encoding
# -----------------------------
model_encoding = {
    "Toyota": 0,
    "Honda": 1,
    "Ford": 2
}

# -----------------------------
# Training Dataset
# -----------------------------
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

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train KNN Model
# -----------------------------
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_scaled, y)

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter Car Details")

model_name = st.selectbox(
    "Car Brand",
    list(model_encoding.keys())
)

model_year = st.number_input(
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

accident = st.selectbox(
    "Accident History",
    ["No", "Yes"]
)

clean_title = st.selectbox(
    "Clean Title",
    ["No", "Yes"]
)

# -----------------------------
# Encode Inputs
# -----------------------------
model_encoded = model_encoding[model_name]

accident_value = 1 if accident == "Yes" else 0
clean_title_value = 1 if clean_title == "Yes" else 0

# -----------------------------
# Prepare Input Data
# -----------------------------
features = np.array([
    [model_encoded, model_year, mileage, engine, accident_value, clean_title_value]
])

features_scaled = scaler.transform(features)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    prediction = knn.predict(features_scaled)

    st.success(f"💰 Predicted Price: ${prediction[0]:,.2f}")

# -----------------------------
# Show Training Data
# -----------------------------
if st.checkbox("Show Training Data"):

    st.subheader("Training Features")
    st.write(X)

    st.subheader("Training Prices")
    st.write(y)
