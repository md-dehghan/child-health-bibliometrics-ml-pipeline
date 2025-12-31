# 📘 Child-Health Relevance Classification Pipeline  
### Machine-learning workflow for classifying bibliometric publications as *child-health relevant* or *not relevant*

This repository contains the full computational pipeline used to classify a large corpus of bibliometric publications into **child-health relevant** vs. **not relevant**, using manual annotations, text preprocessing, TF-IDF vectorization, and supervised machine learning (XGBoost).

---

## 🧭 Overview

The workflow:

1. Processes bibliometric text (titles, abstracts, metadata)  
2. Builds abbreviation dictionaries  
3. Cleans and tokenizes text  
4. Trains a supervised classifier using human-labeled publications  
5. Predicts relevance for the full publication   

---

# 📂 Project Structure

```
project/
├── README.md
├── scripts/
│   ├── make_abbreviation_dicts.py
│   ├── make_processed_text.py
│   ├── train_model.py
│   ├── train_model_hyperparam.py
│   └── predict_labels.py
├── data/
│   ├── raw/              # User-supplied Scopus data (NOT included)
│   ├── interim/          # Abbreviation dictionaries (generated)
│   └── processed/        # Tokenized text + predictions (generated)
├── models/
│   ├── model_output/     # Metrics, ROC, confusion matrix (generated)
│   └── *.pkl             # Trained XGBoost model(s) (generated)
└── Makefile
```

---

# 📢 Data Availability & Scopus Licensing Restrictions

⚠️ **Important notice**

The raw publication metadata used in this pipeline (titles, abstracts, journals, subject areas) were retrieved from **Elsevier’s Scopus API**.

Under **Scopus data-sharing conditions**, researchers **cannot redistribute** files containing Scopus-restricted metadata.

### What **cannot** be shared
❌ `data/raw/Pubs_df.csv`  
❌ `data/raw/Pubs_labeled.csv`  
❌ Any raw metadata from Scopus

If you wish to reproduce the workflow, you must **retrieve the raw metadata yourself** using Scopus

---

# 📦 Data Description

### **`data/raw/`**  
*(Not included — must be created by the user)*

This directory should contain two Scopus-derived CSVs:

- **`Pubs_labeled.csv`**  
  Human-annotated publications with binary relevance labels  
  (`1 = child-health relevant`, `0 = not relevant`)  

- **`Pubs_df.csv`**  
  Publication metadata (title, abstract, journal, etc.)  
  Must be retrieved by the user via Scopus.

---

### **`data/interim/`**  
*(Generated automatically)*

- `abbreviation_dicts_abs.json`
  Detected abbreviations extracted directly from the abstracts during preprocessing.
- `updated_abbreviation_dicts_abs.json`
A refined version of the abbreviation dictionary where abbreviation variants referring to the same concept have been unified.

Stores abbreviation detection output and cleaned mappings.

---

### **`data/processed/`**  
*(Generated automatically)*

- `Pubs_processed_tokens.csv`  
  Tokenized and normalized text 
- `Pubs_with_predictions.csv`  
  Classifier predictions + probabilities
  
---

### **`models/`**  
*(Generated after running training scripts)*

This directory is created when you run the model training scripts.

It contains:

- **`child_health_xgb_pipeline.pkl`**  
  TF-IDF + XGBoost trained pipeline  
- **`model_output/`**  
  Evaluation results:
  - ROC curve  
  - Confusion matrix  
  - Precision/Recall/F1  
  - Classification report  

---

# 🔧 Pipeline Components

## **1. Abbreviation dictionary construction**  
**`scripts/make_abbreviation_dicts.py`**

- Detects abbreviations in abstracts  
- Builds and saves JSON dictionaries  

Outputs:

- `abbreviation_dicts_abs.json`  
- `updated_abbreviation_dicts_abs.json`  
---

## **2. Text processing & tokenization**  
**`scripts/make_processed_text.py`**

- Expands abbreviations  
- Normalizes text  
- Tokenizes titles & abstracts  
- Generates fields used for TF-IDF

Output:

- `data/processed/Pubs_processed_tokens.csv`

---

## **3. Train the classifier**

### A. **Default model**
**`scripts/train_model.py`**

Performs:

- TF-IDF vectorization  
- Train/test split  
- XGBoost training  
- Metrics generation

### B. **Hyperparameter tuning (optional)**  
**`scripts/train_model_hyperparam.py`**

- RandomizedSearchCV   
- Saves best-performing model

Outputs:

- Trained model `.pkl`  
- Metrics in `models/model_output/`

---

## **4. Predict relevance**
**`scripts/predict_labels.py`**

- Loads trained model  
- Predicts relevance for full corpus  
- Saves predictions as CSV

Output:

- `data/processed/Pubs_with_predictions.csv`

---

# 🚀 Running the Pipeline

## **Option 1 — Use the Makefile **

```bash
make
```

Runs all steps in sequence.

Run individual components:

```bash
make abbrev
make process
make train
make train_hyper
make predict
```

---

## **Option 2 — Run scripts manually**


---

# 📊 Model Description

- **Model:** XGBoost (binary classifier)  
- **Features:** TF-IDF vectors from processed title + abstract tokens  
- **Labels:** Human-annotated relevance labels  
- **Metrics:**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1  
  - Confusion matrix  

---

# 📬 Contact

**Masoumeh Dehghani**  
Data Analyst, Alberta Children’s Hospital Research Institute (ACHRI)  
University of Calgary  
📧 masoumeh.dehghanimog@ucalgary.ca | dm.dehghani@gmail.com
