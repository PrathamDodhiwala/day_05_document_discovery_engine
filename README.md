
# Day 5 - Intelligent Document Discovery & Search Engine

A document intelligence application built with Natural Language Processing and unsupervised machine learning.

## Features

- Upload TXT and CSV documents
- Built-in demo document collection
- Text cleaning and preprocessing
- TF-IDF document representation
- Document search and ranking
- Cosine similarity analysis
- Non-negative Matrix Factorization (NMF) topic discovery
- K-Means document clustering
- Related document recommendations
- Document library and inspection
- CSV export of search results

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Cosine Similarity
- NMF
- K-Means
- Streamlit
- Plotly

## Installation

### 1. Create a virtual environment

```powershell
py -3.10 -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip --timeout 600 --retries 10
python -m pip install -r requirements.txt --no-cache-dir --timeout 600 --retries 10
```

### 4. Run the application

```powershell
streamlit run app.py
```

## How it works

### TF-IDF
Converts document text into numerical feature vectors based on term frequency and inverse document frequency.

### Cosine Similarity
Compares document vectors and ranks documents according to their similarity to a search query or another document.

### NMF
Discovers recurring groups of terms that can represent latent topics in a document collection.

### K-Means
Groups documents based on their TF-IDF feature representations.

## Dataset

The application includes a built-in demo collection.

Custom data can be uploaded as:
- TXT files: one document per file.
- CSV files: one row per document, with a text column and an optional title column.

## Important limitation

The search engine uses TF-IDF and cosine similarity. It is not a transformer-based semantic search engine and does not understand context in the same way as an embedding model or LLM.

## Project Objective

Demonstrate practical skills in NLP, information retrieval, text vectorization, similarity analysis, topic modelling and interactive application development.
