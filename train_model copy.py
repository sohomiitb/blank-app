# train_model.py
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

# Dummy data: y = 2*x + 3
X = np.array([[1], [2], [3], [4], [5]])
y = 2 * X.flatten() + 3

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
