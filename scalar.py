from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Example data
X = np.array([[10, 200],
              [20, 300],
              [30, 400]])

# Create scaler
scaler = StandardScaler()

# Fit scaler
scaler.fit(X)

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))