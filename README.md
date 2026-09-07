# 🤖 Text Summarization using BART

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)


## 📌 Project Overview

This project implements an **abstractive text summarization system** using the **BART-Large-CNN Transformer architecture** and the **XSum (Extreme Summarization)** dataset.

The objective is to build a deep learning model capable of reading news articles and generating **highly condensed, meaningful summaries** while preserving the most important information from the original article.

Unlike extractive summarization, which selects existing sentences from a document, abstractive summarization generates new text using a sequence-to-sequence Transformer model.

### 🎯 Main Objective

The project focuses on developing an end-to-end NLP pipeline that:

* Loads and preprocesses the XSum dataset
* Tokenizes news articles and reference summaries
* Fine-tunes a pretrained BART model
* Evaluates the model using ROUGE metrics
* Generates summaries for unseen news articles
* Saves the trained model and tokenizer
* Demonstrates practical deployment possibilities for a Transformer-based NLP model

---

## 🧠 Model Architecture

The project uses:

**BART-Large-CNN (****`facebook/bart-large-cnn`****)**

BART is a Transformer-based sequence-to-sequence architecture designed for natural language generation and other text-to-text tasks.

The `facebook/bart-large-cnn` checkpoint provides a pretrained foundation for abstractive summarization.

### Model Pipeline

```text
                 Input News Article
                       │
                       ▼
              Text Preprocessing
                       │
                       ▼
                 BART Tokenizer
                       │
                       ▼
                  BART Encoder
                       │
                       ▼
              Transformer Layers
                       │
                       ▼
                  BART Decoder
                       │
                       ▼
              Generated Summary
                       │
                       ▼
                ROUGE Evaluation
```

---

## 📊 Dataset

### XSum Dataset

The **XSum (Extreme Summarization)** dataset is a large-scale news summarization dataset designed for highly abstractive summarization.

The dataset contains news articles paired with professionally written summaries.

The main objective is to generate a **very concise summary** of each news article, typically capturing the main point of the article.

### Dataset Structure

Each example contains:

```text
document → News article
summary  → Reference summary
```

### Example

**News Article**

```text
The government announced a new economic policy today
following several months of discussions with industry
representatives. Officials said the changes would help
businesses increase investment and create new jobs.
```

**Reference Summary**

```text
Government announces new economic policy.
```

The model learns the relationship:

```text
News Article → Extreme Abstractive Summary
```

The dataset is particularly useful for training models to summarize:

* News articles
* Online journalism
* Long-form news content
* Current-affairs articles
* Media content

---

## ✨ Key Features

* 🔤 Natural Language Processing
* 🤖 Transformer-based deep learning
* 🧠 BART pretrained model
* 📰 News article summarization
* 📚 XSum dataset
* 🎯 Abstractive text generation
* 📏 ROUGE-based evaluation
* 🚀 GPU acceleration
* 🧪 Model evaluation
* 💾 Model persistence
* 📝 Custom summary generation

---

# 🛠️ Technologies Used

| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| Python                    | Programming language            |
| PyTorch                   | Deep learning framework         |
| Hugging Face Transformers | BART implementation             |
| Hugging Face Datasets     | Dataset loading and processing  |
| Evaluate                  | Model evaluation                |
| ROUGE                     | Summarization evaluation        |
| Accelerate                | Training optimization           |
| Google Colab              | Development and GPU environment |
| Jupyter Notebook          | Experimentation                 |

---

# 📁 Project Structure

```text
bart-xsum-text-summarization/
│
├── README.md
│
├── notebooks/
│   └── BART_XSum_Text_Summarization.ipynb
│
├── requirements.txt
│
└── .gitignore
```

> The exact folder structure can be modified depending on how the project is organized in your GitHub repository.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bart-xsum-text-summarization.git
```

Move into the project directory:

```bash
cd bart-xsum-text-summarization
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
transformers
datasets
evaluate
rouge_score
accelerate
numpy
pandas
tqdm
sentencepiece
```

---

# 🚀 Running the Project

## Option 1 — Google Colab

The recommended way to run this project is using Google Colab with GPU acceleration.

Open the notebook:

```text
notebooks/BART_XSum_Text_Summarization.ipynb
```

Then:

1. Open the notebook in Google Colab.
2. Enable GPU.
3. Install the required libraries.
4. Verify GPU availability.
5. Load the XSum dataset.
6. Explore the dataset.
7. Preprocess the articles and summaries.
8. Tokenize the dataset.
9. Load the BART model.
10. Fine-tune BART.
11. Evaluate the model.
12. Generate summaries.
13. Save the model and tokenizer.

---

# 🔧 Hardware Requirements

Training Transformer models can be computationally expensive.

### Recommended

```text
GPU: NVIDIA T4 / RTX 3060 or better
RAM: 12 GB+
Storage: 10 GB+
Python: 3.9+
```

Google Colab GPU is sufficient for experimentation and educational purposes.

For larger experiments, a higher-memory GPU can provide better training performance.

---

# 📚 Methodology

The project follows the following machine learning pipeline:

```text
XSum Dataset
     │
     ▼
Data Exploration
     │
     ▼
Data Preprocessing
     │
     ▼
Tokenization
     │
     ▼
Train / Validation / Test
     │
     ▼
BART Fine-Tuning
     │
     ▼
Model Evaluation
     │
     ▼
ROUGE Metrics
     │
     ▼
Summary Generation
     │
     ▼
Model Saving
```

---

# 1️⃣ Environment Setup

The first stage installs and imports the required libraries.

Main libraries include:

```python
import torch
import pandas as pd
import numpy as np

from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    TrainingArguments,
    Trainer
)
```

The availability of a GPU is also checked before training.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
```

The NVIDIA GPU can also be inspected using:

```bash
nvidia-smi
```

---

# 2️⃣ Dataset Loading

The XSum dataset is loaded using Hugging Face Datasets.

```python
from datasets import load_dataset

dataset = load_dataset("EdinburghNLP/xsum")
```

The dataset contains:

```text
train
validation
test
```

Each example contains:

```text
document
summary
```

The `document` is used as the model input, while the `summary` is used as the target output.

---

# 3️⃣ Data Exploration

Before training, the dataset can be inspected.

```python
print(dataset["train"][0])
```

An individual article and its reference summary can be displayed using:

```python
article = dataset["train"][0]["document"]
summary = dataset["train"][0]["summary"]

print("Article:")
print(article)

print("\nReference Summary:")
print(summary)
```

This step helps verify that the dataset has been loaded correctly.

---

# 4️⃣ Data Preprocessing

The news article and reference summary are prepared for the Transformer model.

The input is:

```text
Document
```

The target is:

```text
Summary
```

The model learns:

```text
Document → Summary
```

Very long documents may need to be truncated because BART has a maximum input sequence length.

---

# 5️⃣ Tokenization

BART requires text to be converted into numerical token IDs.

The pretrained tokenizer is loaded using:

```python
from transformers import BartTokenizer

model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name)
```

The input articles are limited to **1024 tokens**, while the summaries are limited to approximately **128 tokens**.

Example preprocessing function:

```python
def tokenize_function(examples):

    model_inputs = tokenizer(
        examples["document"],
        max_length=1024,
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["summary"],
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
```

The preprocessing function can then be applied to the dataset:

```python
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)
```

---

# 6️⃣ Model Initialization

The pretrained BART model is loaded using:

```python
from transformers import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn"
)
```

The pretrained model provides a strong starting point for abstractive summarization.

The model is then fine-tuned using the XSum dataset.

---

# 7️⃣ Model Training

During training, BART learns the relationship between:

```text
News Article → Summary
```

The model updates its parameters using the training data to minimize the sequence-to-sequence loss.

A GPU is used whenever available to accelerate training.

Example training configuration:

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2
)
```

The Hugging Face `Trainer` API can be used to train the model:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)
```

Training is started using:

```python
trainer.train()
```

---

# 8️⃣ Model Evaluation

The trained model is evaluated using unseen data.

The primary evaluation metrics are:

* ROUGE-1
* ROUGE-2
* ROUGE-L

These metrics compare the generated summaries against the human-written reference summaries.

---

# 📏 ROUGE Metrics

## ROUGE-1

Measures the overlap of individual words between the generated summary and reference summary.

## ROUGE-2

Measures the overlap of two-word sequences, known as bigrams.

## ROUGE-L

Measures the longest common subsequence between the generated summary and reference summary.

Higher ROUGE scores generally indicate greater similarity to the reference summaries.

However, ROUGE alone does not completely measure factual correctness or summary quality.

---

# 📊 Evaluation Results

The current experiment achieved a **baseline ROUGE-1 score of approximately 0.014** on a small evaluation sample.

| Metric  |              Score |
| ------- | -----------------: |
| ROUGE-1 | **0.014302**       |
| ROUGE-2 | **0.000000**       |
| ROUGE-L | **0.014156**       |

### Result Interpretation

The current score should be considered an **initial baseline** rather than the final performance of the BART model.

The relatively low score may be influenced by:

* Small evaluation sample
* Limited training duration
* One training epoch
* Training dataset size
* Generation configuration
* Tokenization configuration
* Evaluation implementation

Further experiments using larger training and evaluation subsets can provide a more reliable measurement of model performance.

---

# 🧪 Example Prediction

After training, the model can generate summaries for unseen news articles.

### Input

```text
The government announced a new economic policy today
following several months of discussions with industry
representatives. Officials said the changes would help
businesses increase investment and create new jobs.
```

### Reference Summary

```text
Government announces new economic policy.
```

### Generated Summary

```text
The government has announced a new economic policy.
```

> The generated output will depend on the trained model checkpoint and generation parameters.

---

# 🔮 Custom Prediction

The trained BART model can be used to summarize new news articles.

Example:

```python
article = """
The government announced a new economic policy today
following several months of discussions with industry
representatives. Officials said the changes would help
businesses increase investment and create new jobs.
"""

inputs = tokenizer(
    article,
    return_tensors="pt",
    max_length=1024,
    truncation=True
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=128,
    min_length=10,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("Generated Summary:")
print(summary)
```

---

# 💾 Model Persistence

After training, the model and tokenizer can be saved for future inference.

```python
model.save_pretrained(
    "./model/bart-xsum-finetuned"
)

tokenizer.save_pretrained(
    "./model/bart-xsum-finetuned"
)
```

The saved model can later be loaded using:

```python
model = BartForConditionalGeneration.from_pretrained(
    "./model/bart-xsum-finetuned"
)

tokenizer = BartTokenizer.from_pretrained(
    "./model/bart-xsum-finetuned"
)
```

---

# 📈 Performance Analysis

The performance of the model depends on several factors:

* Dataset size
* Training epochs
* Learning rate
* Batch size
* Gradient accumulation
* Maximum input length
* Maximum output length
* GPU availability
* Beam search configuration
* Quality of preprocessing

Fine-tuning for additional epochs may improve performance, but excessive training can increase training time and potentially lead to overfitting.

A useful future experiment is to compare:

```text
1 Epoch
   ↓
2 Epochs
   ↓
3 Epochs
```

and record:

```text
ROUGE-1
ROUGE-2
ROUGE-L
```

for each experiment.

---

# ⚠️ Limitations

Although BART is effective for abstractive summarization, this project has several limitations.

### 1. Computational Requirements

BART-Large requires significant computational resources, particularly during fine-tuning.

### 2. Input Length

Very long news articles may exceed the maximum input sequence length and therefore require truncation.

### 3. Hallucination

The model may occasionally generate information that is not explicitly supported by the original article.

### 4. Training Time

Fine-tuning a large Transformer model can take considerable time without GPU acceleration.

### 5. Extreme Summarization

XSum encourages highly condensed summaries. As a result, some important information from the original article may not appear in the generated summary.

### 6. Evaluation Limitations

ROUGE primarily measures text overlap and does not completely capture factual consistency, semantic quality, or usefulness.

---

# 🔐 Ethical Considerations

Automated news summarization systems should be used carefully.

Potential concerns include:

* Incorrect information
* Hallucinated facts
* Loss of important context
* Bias in generated summaries
* Misrepresentation of news content
* Over-reliance on automatically generated summaries

For high-impact applications, generated summaries should be reviewed against the original source article.

---

# 🚀 Future Improvements

Several improvements can be made to this project.

### 🔹 Model Improvements

* Compare BART with PEGASUS
* Compare BART with T5
* Experiment with larger sequence-to-sequence models
* Experiment with modern instruction-tuned models

### 🔹 Training Improvements

* Increase training epochs
* Train using larger portions of the XSum dataset
* Perform hyperparameter tuning
* Use learning-rate scheduling
* Experiment with different batch sizes
* Apply gradient accumulation
* Experiment with mixed-precision training

### 🔹 Evaluation Improvements

Add additional metrics such as:

* BERTScore
* BLEURT
* METEOR

Human evaluation can also be introduced to measure:

* Fluency
* Relevance
* Factual consistency
* Readability
* Information coverage

### 🔹 Error Analysis

A dedicated error-analysis pipeline can be added:

```text
Input Article
      ↓
Reference Summary
      ↓
Generated Summary
      ↓
Error Analysis
```

Potential errors include:

* Missing information
* Incorrect facts
* Repetition
* Hallucination
* Overly generic summaries
* Poor sentence structure

### 🔹 Deployment

The model could be deployed using:

```text
FastAPI
      ↓
REST API
      ↓
BART Model
      ↓
Generated Summary
```

A web application could also be developed using:

```text
React / HTML / CSS
        ↓
FastAPI
        ↓
BART
        ↓
Summary
```

---

# 🌐 Possible Real-World Applications

This project can be extended to applications such as:

* 📰 News summarization
* 📱 News aggregation
* 🗞️ Media monitoring
* 📧 Newsletter summarization
* 📊 Business intelligence
* 🔎 Search-result summarization
* 📚 Document summarization
* 🤖 AI-powered information assistants

---

# 🎓 Learning Outcomes

Through this project, the following concepts are demonstrated:

### Machine Learning

* Dataset preparation
* Training and validation
* Model evaluation
* Prediction
* Experimentation

### Deep Learning

* Neural networks
* Transformer architecture
* Encoder-decoder models
* Sequence-to-sequence learning
* Transfer learning
* Fine-tuning pretrained models

### NLP

* Text preprocessing
* Tokenization
* Abstractive summarization
* Text generation
* Sequence modeling
* Natural language evaluation

### MLOps / Engineering

* Environment management
* GPU acceleration
* Model saving
* Evaluation pipelines
* Reproducible experiments
* GitHub project organization

---

# 🧩 Challenges Faced

During development, several challenges may occur.

### Dataset and Preprocessing

News articles vary considerably in length and structure, requiring appropriate preprocessing and truncation.

### Transformer Memory Usage

BART-Large can require substantial GPU memory, particularly when processing long input sequences.

### Evaluation Libraries

Different versions of the Hugging Face evaluation ecosystem may require different configurations.

### Training Performance

Training speed depends heavily on GPU hardware, batch size, sequence length, and gradient accumulation.

### Generation Quality

Generation parameters such as beam size, maximum length, and minimum length can significantly affect summary quality.

---

# 🛠️ Troubleshooting

## CUDA / GPU Not Available

Check:

```python
torch.cuda.is_available()
```

If it returns:

```text
False
```

the notebook is running on CPU.

In Google Colab, enable:

```text
Runtime → Change runtime type → GPU
```

---

## Out of Memory Error

Reduce the batch size:

```python
per_device_train_batch_size=2
```

or:

```python
per_device_train_batch_size=1
```

You can also increase gradient accumulation:

```python
gradient_accumulation_steps=4
```

Reducing the maximum input length can also decrease GPU memory usage.

---

## Metric Loading Error

If an evaluation metric fails to load, install the required packages:

```bash
pip install evaluate rouge_score
```

Then restart the runtime if necessary.

---

## Model Generation Problems

If the model generates empty, repetitive, or unexpectedly short summaries, check:

* Tokenization
* `max_length`
* `min_length`
* Beam-search parameters
* Model checkpoint
* Input formatting
* Training quality

---

# 📦 Requirements

Recommended Python version:

```text
Python 3.9+
```

Required packages:

```text
torch
transformers
datasets
evaluate
rouge_score
accelerate
numpy
pandas
tqdm
sentencepiece
```

Install everything with:

```bash
pip install -r requirements.txt
```

---

# 📜 License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this project according to the terms of the license.

---

# 👨‍💻 Author

**Sudeera Attanayake**

AI Engineer

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Natural Language Processing
* Agentic AI
* Generative AI
* MLOps

---

# ⭐ Acknowledgements

Special thanks to the open-source communities behind:

* Hugging Face
* PyTorch
* Meta AI / Facebook AI
* XSum dataset contributors
* Google Colab

This project was developed for educational and portfolio purposes.

---

# 📚 References

* BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
* XSum: New Extreme Summarization Dataset for News Summarization
* Hugging Face Transformers documentation
* Hugging Face Datasets documentation
* ROUGE evaluation methodology

---

# ⭐ Project Highlights

```text
┌─────────────────────────────────────────────┐
│          BART TEXT SUMMARIZATION            │
├─────────────────────────────────────────────┤
│                                             │
│  Dataset       → XSum                       │
│  Architecture  → Transformer / BART         │
│  Task          → Abstractive Summarization  │
│  Domain        → News Articles              │
│  Framework     → PyTorch                    │
│  NLP Library   → Hugging Face Transformers  │
│  Evaluation    → ROUGE                      │
│  Environment   → Google Colab / Python      │
│                                             │
└─────────────────────────────────────────────┘
```

---

# 📌 Conclusion

This project demonstrates the complete development workflow for an **abstractive news summarization system using BART and the XSum dataset**.

By fine-tuning the pretrained `facebook/bart-large-cnn` Transformer model, the system learns to transform news articles into highly condensed summaries.

The project provides practical experience with:

```text
NLP
 ↓
Dataset Processing
 ↓
Tokenization
 ↓
Transformer Models
 ↓
Fine-Tuning
 ↓
Evaluation
 ↓
Text Generation
 ↓
Model Persistence
 ↓
Deployment Possibilities
```

The current experiment provides an initial baseline for future optimization. Further training with larger data subsets, additional epochs, improved generation strategies, and broader evaluation can be used to improve the system.

This project can serve as a foundation for developing advanced NLP applications such as automated news aggregation, media monitoring, document summarization, and AI-powered information assistants.

---

## ⭐ If you found this project useful

Give this repository a ⭐ on GitHub and feel free to fork the project and experiment with different Transformer architectures, training strategies, and summarization datasets.
