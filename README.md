
# Fine-Tuning Phi-2 for Disease Classification using LORA

This notebook demonstrates the process of **fine-tuning Microsoft's Phi-2** model for **binary classification** of disease-related abstracts using **LoRA (Low-Rank Adaptation)**. It involves both baseline evaluation and fine-tuned comparison, and it uses structured biomedical abstract data to predict whether the text indicates "Cancer" or "Non Cancer".

---

## Project Structure

- `data/`: Contains raw biomedical abstract files labeled as "cancer" and "non_cancer".
- `checkpoints/`: Stores LoRA-adapted model weights.
- `Fine_Tunning_Part_2.ipynb`: Main notebook for preprocessing, training, inference, and evaluation.

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

## 📊 Output

- Model predictions: Full probability distributions per abstract.
- Extracted disease mentions using few-shot prompting.
- Useful for benchmarking downstream biomedical classification tasks.

---

## 💡 Future Enhancements

- Add evaluation metrics (accuracy, F1, ROC-AUC).
- Incorporate zero-shot vs fine-tuned comparison.
- Use structured output for disease mention extraction (NER or span-based tagging).
