import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Title
# -----------------------------
st.title("🚗 Car Price Predictor (KNN Model)")

# -----------------------------
# Example training data
# Features: [model_encoded, model_year, mileage, engine, accident, clean_title]
# -----------------------------
model_encoding = {
    "Toyota": 0,
    "Honda": 1,
    "Ford": 2
}

X = np.array([
    [0, 2015, 50000, 1500, 0, 1],
    [1, 2017, 30000, 1600, 0, 1],
    [2, 2012, 80000, 1400, 1, 0],
    [0, 2020, 10000, 2000, 0, 1],
    [1, 2018, 40000, 1800, 1, 1],
])

y = np.array([
    15000,
    18000,
    9000,
    25000,
    20000
])

# -----------------------------
# Feature Scaling (important for KNN)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_scaled, y)

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter Car Details")

model_name = st.selectbox(
    "Model Name",
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
    value=50000
)

engine = st.number_input(
    "Engine Size (cc)",
    min_value=500,
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

# Convert categorical values
accident = 0 if accident == "No" else 1
clean_title = 0 if clean_title == "No" else 1
model_encoded = model_encoding[model_name]

# Prepare input features
features = np.array([
    [model_encoded, model_year, mileage, engine, accident, clean_title]
])

# Scale features
features_scaled = scaler.transform(features)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    prediction = knn.predict(features_scaled)

    st.success(f"💰 Predicted Price: ${prediction[0]:,.2f}")

# -----------------------------
# Show training data
# -----------------------------
if st.checkbox("Show Training Data"):
    st.subheader("Training Features")
    st.write(X)

    st.subheader("Training Prices")
    st.write(y)

    st.write("Training Targets (Price):")
    st.write(y)
