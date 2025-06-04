import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Generate sample dataset
np.random.seed(42)
n = 100
sqft = np.random.randint(500, 4000, size=n)
bedrooms = np.random.randint(1, 6, size=n)
price = 50000 + sqft * 200 + bedrooms * 10000 + np.random.normal(0, 10000, size=n)

df = pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "price": price
})

X = df[["sqft", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "price_model.pkl")
print("✅ Model saved as 'price_model.pkl'")
