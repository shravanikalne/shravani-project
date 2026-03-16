import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


def main():

    # ----------------------
    # Sample Dataset
    # ----------------------
    data = {
        "model": [0, 1, 2, 0, 1],
        "year": [2015, 2017, 2012, 2020, 2018],
        "mileage": [50000, 30000, 80000, 10000, 40000],
        "engine": [1500, 1600, 1400, 2000, 1800],
        "accident": [0, 0, 1, 0, 1],
        "clean_title": [1, 1, 0, 1, 1],
        "price": [15000, 18000, 9000, 25000, 20000]
    }

    df = pd.DataFrame(data)

    # ----------------------
    # Features and Target
    # ----------------------
    X = df.drop("price", axis=1)
    y = df["price"]

    # ----------------------
    # Train-Test Split
    # ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------
    # Feature Scaling
    # ----------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------
    # Train KNN Model
    # ----------------------
    k = 3
    model = KNeighborsRegressor(n_neighbors=k)

    model.fit(X_train_scaled, y_train)

    # ----------------------
    # Save Model and Scaler
    # ----------------------
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    print("Model trained successfully")
    print("model.pkl and scaler.pkl saved")


if __name__ == "__main__":
    main()main()
    
