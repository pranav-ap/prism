# Clear News

A modular NLP system for analyzing and critiquing news articles using large language models. It consists of:

* 🗳️ **Political Bias Classifier** — classifies articles as `left`, `center`, or `right` leaning
* ❓ **Critic Generator** — generates probing or critical questions based on the content

## 🗳️ Political Bias Classifier

* Model: `bert-base-uncased` with LoRA fine-tuning
* Quantized to 4-bit (NF4) using `bitsandbytes`
* Trained on [
  `article-bias-prediction-media-splits`](https://huggingface.co/datasets/siddharthmb/article-bias-prediction-media-splits)
* 3 classes: `left`, `center`, `right`

### 🌐 API: Classifier

**POST** `/predict`

```json
{ "text": "The government introduced a universal healthcare plan." }
```

**Response:**

```json
{ "bias": "left" }
```

Run locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## ❓ Critic Generator

### 🔍 Description

* Model: Quantized LLaMA 3.2B / Mistral (4-bit with LoRA)
* Task: Generate critical questions based on news content
* Dataset: Custom or argumentative datasets with (text → question) pairs

### 🚀 Training

```bash
cd critic_generator
python train.py
```

Uses Hugging Face `Trainer`, LoRA + QLoRA for low-RAM training.

### 🧪 Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("checkpoints/")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/")

prompt = "TEXT: The senator refused to sign the climate bill. QUESTION:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(output[0]))
```

---

### 🌐 API: Critic Generator

**POST** `/generate-question`

```json
{ "text": "The senator refused to sign the climate bill." }
```

**Response:**

```json
{ "question": "Why did the senator oppose the bill?" }
```

Run locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

## 🧩 Future Work

* Add more nuanced classifications
* Generate specific kinds of counter arguments
* Add authentication to the API
