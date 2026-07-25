# 🧠 Smart MCQ Solver

An AI-powered Multiple Choice Question Answering system that predicts the **Top-3 most probable answers** for challenging MCQs using deep learning, transformer models, and Retrieval-Augmented Generation (RAG).

Built for the **Smart MCQ Solver Challenge**, where submissions are evaluated using **Mean Average Precision at 3 (MAP@3)**.

Try it out on [Hugging Face Spaces](https://huggingface.co/spaces/dhruvbansalup/dl-genai-project-26-t2-smart-mcq-solver).

---

## 📌 Overview

Smart MCQ Solver is an experimental framework for solving multiple-choice questions using modern Natural Language Processing techniques.

The project explores multiple approaches ranging from classical machine learning models to state-of-the-art transformer architectures and Retrieval-Augmented Generation (RAG), comparing their performance on the same benchmark.

The objective is not only to predict the correct answer but to rank the **three most likely answers**, maximizing the MAP@3 evaluation metric.

---

## ✨ Features

- Classical ML baseline
- Custom BiLSTM model
- Fine-tuned Transformer models
  - BERT
  - RoBERTa
  - DeBERTa-v3
- Retrieval-Augmented Generation (RAG)
- FAISS vector search
- Cross-Encoder reranking
- PyTorch Lightning training pipeline
- Weights & Biases experiment tracking
- Automatic inference pipeline
- Kaggle submission generation

---

## 📂 Dataset

Each sample consists of:

| Column | Description |
|---------|-------------|
| id | Question ID |
| prompt | Question statement |
| A-E | Five answer choices |
| answer | Correct option (training only) |

The test dataset excludes the `answer` column, requiring models to predict the top three most probable answer labels. :contentReference[oaicite:2]{index=2}

---

## 📊 Evaluation Metric

The competition evaluates submissions using:

> **Mean Average Precision @ 3 (MAP@3)**

The model predicts three ranked answer labels.

Example:

```
Correct Answer:
A

Prediction:
A B C   ✅ Highest score
B A C   ✅ Lower score
C D A   ✅ Lowest score
```

---

## 🤖 Models Implemented

### 1. TF-IDF + LightGBM
A classical machine learning baseline using TF-IDF features with a LightGBM classifier.

---

### 2. BiLSTM

A custom Bidirectional LSTM model trained on tokenized MCQ text.

Features:

- Embedding Layer
- Bidirectional LSTM
- Dropout
- Fully Connected Classifier

---

### 3. Transformer Models

Fine-tuned HuggingFace transformers:

- BERT (`bert-base-uncased`)
- RoBERTa (`roberta-base`)
- DeBERTa-v3 (`microsoft/deberta-v3-base`)

Training performed using PyTorch Lightning.

---

### 4. Retrieval-Augmented Generation (RAG)

The RAG pipeline enhances transformer predictions using external context retrieved from the training corpus.

Pipeline:

```
Question
      │
Sentence Transformer
      │
Vector Embedding
      │
FAISS Retrieval
      │
Top-K Relevant Context
      │
Cross Encoder Re-ranking
      │
Context Injection
      │
Transformer Prediction
```

Components:

- SentenceTransformer embeddings
- FAISS vector search
- CrossEncoder reranker
- Transformer classifier

---

## 🛠 Tech Stack

- Python
- PyTorch
- PyTorch Lightning
- HuggingFace Transformers
- Sentence Transformers
- FAISS
- Scikit-learn
- LightGBM
- Pandas
- NumPy
- Weights & Biases

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/dhruvbansalup/dl-genai-project-26-t2-Smart_MCQ_Solver.git

cd smart-mcq-solver
```

Install dependencies via conda environment.yml

```bash
conda env create -f environment.yml
conda activate mcq_solver
```

---

## 🏃 Training

```bash
# Training TFIDF-LGBM
python -m src.training.train_classical_tfidf_lgbm --log True --uploadToKaggle True

# Training Custom BiLSTM
python -m src.training.train_bilstm --log False --uploadToKaggle False

# Training pre-trained transformers
python -m src.training.train_transformer --model BERT --log True --uploadToKaggle True
python -m src.training.train_transformer --model RoBERTa --log True --uploadToKaggle True
python -m src.training.train_transformer --model DeBERTa_v3_base --log True --uploadToKaggle True

# Training pre-trained transformers with data_v002
python -m src.training.train_transformer_data_v002 --model BERT --log True --uploadToKaggle True
python -m src.training.train_transformer_data_v002 --model RoBERTa --log True --uploadToKaggle True
python -m src.training.train_transformer_data_v002 --model DeBERTa_v3_base --log True --uploadToKaggle True

# Training pre-trained transformers using RAG pipeline
python -m src.training.train_transformer_with_rag --model BERT --log True --uploadToKaggle True
python -m src.training.train_transformer_with_rag --model RoBERTa --log True --uploadToKaggle True
python -m src.training.train_transformer_with_rag --model DeBERTa_v3_base --log True --uploadToKaggle True --batch_size 2

# Training pre-trained transformers using RAG pipeline with data_v002
python -m src.training.train_transformer_with_rag_data_v002 --model BERT --log True --uploadToKaggle True
python -m src.training.train_transformer_with_rag_data_v002 --model RoBERTa --log True --uploadToKaggle True
python -m src.training.train_transformer_with_rag_data_v002 --model DeBERTa_v3_base --log True --uploadToKaggle True --batch_size 2
```

---

## 🔍 Inference

```bash
# Infer TFIDF-LGBM
python -m src.inference.infer_classical_tfidf_lgbm

# Infer BiLSTM01
python -m src.inference.infer_bilstm

# Infer Zero-Shot bart-large-mnli
python -m src.inference.infer_zero_shot_bart_large_mnli

# Infer pre-trained transformers
python -m src.inference.infer_transformer --model BERT
python -m src.inference.infer_transformer --model RoBERTa
python -m src.inference.infer_transformer --model DeBERTa_v3_base

# Infer Ensemble - BERT+RoBERTa+DeBERTa
python -m src.inference.infer_ensemble

# Infer pre-trained transformers using RAG pipeline
python -m src.inference.infer_transformer_with_rag --model BERT_RAG
python -m src.inference.infer_transformer_with_rag --model RoBERTa_RAG
python -m src.inference.infer_transformer_with_rag --model DeBERTa_v3_base_RAG
```

This generates

```
submission.csv
```

formatted for Kaggle submission.
---

## 📜 License

This project is intended for educational and research purposes.

---