
# Fine-Tuning Phi-2 for Disease Classification using LORA

This notebook demonstrates the process of **fine-tuning Microsoft's Phi-2** model for **binary classification** of disease-related abstracts using **LoRA (Low-Rank Adaptation)**. It involves both baseline evaluation and fine-tuned comparison, and it uses structured biomedical abstract data to predict whether the text indicates "Cancer" or "Non Cancer".

---

## Project Structure

- `data/`: Contains raw biomedical abstract files labeled as "cancer" and "non_cancer".
- `checkpoints/`: Stores LoRA-adapted model weights.
- `Phi_2_Finetuned_Pipeline.ipynb`: Main notebook for preprocessing, training, inference, and evaluation.

---

## Components & Workflow

### 1. **Environment Setup**

Imports necessary Python modules, primarily:
- `transformers`, `peft`, `datasets` – for LLM and LoRA setup.
- `sklearn`, `pandas`, `tqdm`, and `torch` – for preprocessing and evaluation.

### 2. **Base Model Initialization**

- Loads `microsoft/phi-2` for both **causal LM** and **sequence classification** tasks.
- Loads tokenizer and model architecture required for fine-tuning and inference.

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = PhiForSequenceClassification.from_pretrained(model_name)
```

---

### 3. **Data Loading & Parsing**

- Iteratively reads raw biomedical abstracts from file system.
- Segments content into structured key-value dictionaries.
- Converts each dictionary into a row of a `DataFrame`.

> Data is split into **Cancer** and **Non Cancer** categories, both annotated with appropriate class labels.

---

### 4. **Dataset Construction**

- Combines both labeled `DataFrames`.
- Extracts relevant columns like **Abstract** (renamed to `text`) and **Title**.
- Constructs a unified `diagnostic_df` with class labels.

---

### 5. **Model Fine-Tuning with LoRA**

- Loads a pre-trained checkpoint using PEFT for inference.
- Injects LoRA adapters into base model layers to enable parameter-efficient tuning.

```python
model = PeftModel.from_pretrained(base_model, checkpoint_path)
```

---

### 6. **Pipeline Inference – Baseline & Fine-Tuned**

- Constructs a `text-classification` pipeline using both:
  - **Base (unmodified)** model
  - **LoRA-adapted fine-tuned** model

```python
pipeline("text-classification", model=model, tokenizer=tokenizer)
```

- Evaluates performance by feeding test abstract texts and collecting prediction scores.

---

### 7. **Entity Extraction Task**

- Applies LLM prompting to extract **disease names** using generative responses.
- Appends extracted results to a list for evaluation or downstream use.

```python
disease_extraction(f"Explain the disease name only from the abstract text {abstract_extract}")
```

---

## Output

- Model predictions: Full probability distributions per abstract.
- Extracted disease mentions using few-shot prompting.
- Useful for benchmarking downstream  classification tasks.

---

## Evaluation
- Add evaluation metrics (confusion matrix,accuracy, Prcision ,Recall,F1,).
- Incorporate zero-shot vs fine-tuned comparison.
- Use structured output for disease mention extraction (NER or span-based tagging).

#Deployment
VLLM (Very Large Language Model) is an inference engine optimized for fast, efficient serving of LLMs.
Compared to vanilla Hugging Face transformers inference, it offers:

-PagedAttention → reduces memory usage & avoids out-of-memory errors for long prompts.
-Streaming support → sends partial results as they’re generated.

-Easy deployment → run one command and expose a REST API.

For a classification model like Cancer / Non-Cancer detection, VLLM might feel like overkill, but:

-It ensures low-latency inference.

-Can scale easily if you later integrate with other LLM tasks.

-You can serve LoRA / fine-tuned versions of base models.
```
python -m vllm.entrypoints.api_server \
  --model /home/ubuntu/dev_large_model/phi2_cls/checkpoint-300/ \
  --dtype float16 \
  --port=8082
```
python -m vllm.entrypoints.api_server

Starts the VLLM REST API server.

--model /home/ubuntu/dev_large_model/phi2_cls/checkpoint-300/

Path to your model checkpoint (Phi-2 fine-tuned for classification).

Can also be a Hugging Face model name.

--dtype float16

Loads model weights in half precision.

Speeds up inference and reduces GPU memory usage.

--port=8082

REST API will be available at http://localhost:8082.

## Fast API based inferencing

```
from fastapi import FastAPI
import requests
from pydantic import BaseModel
VLLM_ENDPOINT = "http://localhost:8082/generate"  # matches your --port

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

```
## How it Works
Run VLLM Server → loads your fine-tuned Phi-2 classification model into GPU memory.

FastAPI Client Endpoint (/classify):

Receives input text (e.g., "This patient has a malignant tumor").

Sends it to VLLM’s /generate endpoint.

VLLM generates the output (e.g., "Cancer" or "Non-Cancer").

Post-process Output → removes the original prompt from the response if the model repeats it.


