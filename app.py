# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("model.pkl")

# Define request format
class InputData(BaseModel):
    x: float

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.post("/predict")
def predict(data: InputData):
    x_input = np.array([[data.x]])
    y_pred = model.predict(x_input)
    return {"prediction": float(y_pred[0])}
