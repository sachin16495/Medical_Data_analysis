from fastapi import FastAPI
import requests
from pydantic import BaseModel

# VLLM API endpoint
VLLM_ENDPOINT = "http://localhost:8082/classify_abstract"  # matches your --port

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/classify")
def classify_text(input_data: InputText):
    payload = {
        "prompt": input_data.text,
        "temperature": 0.0,   # deterministic output
        "max_tokens": 10      # small since classification needs short answer
    }
    
    # Call VLLM API
    response = requests.post(VLLM_ENDPOINT, json=payload)
    result = response.json()

    # VLLM returns: {"text": ["<prompt><output>"]}
    generated_text = result["text"][0]

    # Remove the prompt from output if the model echoes it
    output_label = generated_text.replace(input_data.text, "").strip()
    
    return {"label": output_label}
