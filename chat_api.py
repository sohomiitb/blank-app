from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
import re
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small LLM
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load trained regression model
price_model = joblib.load("price_model.pkl")

class ChatInput(BaseModel):
    prompt: str

def extract_features(text):
    sqft_match = re.search(r"(\d+)\s*(sqft|square feet|square foot)", text.lower())
    bed_match = re.search(r"(\d+)\s*(bedroom|bed)", text.lower())

    sqft = int(sqft_match.group(1)) if sqft_match else None
    bedrooms = int(bed_match.group(1)) if bed_match else None
    return sqft, bedrooms

@app.post("/chat")
def chat(data: ChatInput):
    prompt = data.prompt.strip()

    # Check if it's an ML query
    sqft, bedrooms = extract_features(prompt)

    if sqft and bedrooms:
        x = np.array([[sqft, bedrooms]])
        pred = price_model.predict(x)[0]
        return {
            "response": f"🧮 Estimated price for a {bedrooms}-bedroom house with {sqft} sqft is around ${pred:,.0f}."
        }

    # Otherwise, fallback to LLM
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = full_output[len(prompt):].strip()
    return {"response": response}
