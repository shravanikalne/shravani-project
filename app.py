import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsRegressor  # <- Use Regressor, not Classifier

# Title
st.title("Car Price Predictor (KNN)")

# Example training data (dummy numeric data)
# Features: [model_encoded, model_year, mileage, engine, accident, clean_title]
# model_encoded: numeric code for each model
model_encoding = {"Toyota": 0, "Honda": 1, "Ford": 2}

X = np.array([
    [0, 2015, 50000, 1500, 0, 1],
    [1, 2017, 30000, 1600, 0, 1],
    [2, 2012, 80000, 1400, 1, 0],
    [0, 2020, 10000, 2000, 0, 1],
    [1, 2018, 40000, 1800, 1, 1],
])
y = np.array([15000, 18000, 9000, 25000, 20000])  # Example prices

# Train KNN regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)

# User inputs
st.header("Enter Car Details")

model_name = st.selectbox("Model Name", list(model_encoding.keys()))
model_year = st.number_input("Model Year", min_value=1990, max_value=2026, value=2018)
mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
engine = st.number_input("Engine (cc)", min_value=500, value=1500)
accident = st.selectbox("Accident History", ["No", "Yes"])
accident = 0 if accident == "No" else 1
clean_title = st.selectbox("Clean Title", ["No", "Yes"])
clean_title = 0 if clean_title == "No" else 1

# Map model name to numeric code
model_encoded = model_encoding[model_name]

# Prepare feature vector
features = np.array([[model_encoded, model_year, mileage, engine, accident, clean_title]])

# Predict button
if st.button("Predict Price"):
    prediction = knn.predict(features)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")

# Optional: show training data
if st.checkbox("Show Training Data"):
    st.write("Training Features:")
    st.write(X)
    st.write("Training Targets (Price):")
    st.write(y)