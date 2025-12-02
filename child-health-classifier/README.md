# 📘 Child-Health Relevance Classification Pipeline  
### ML workflow for classifying bibliometric publications as *child-health relevant* or *not relevant*

This repository contains the full computational pipeline used to classify a large corpus of bibliometric publications into **child-health relevant** vs. **not relevant**, using manually annotated data, text preprocessing, and supervised machine learning.

---

## 🧭 Overview

The pipeline processes raw bibliometric data (titles, abstracts, metadata), generates abbreviation dictionaries, normalizes text, trains a classifier using labeled data, and predicts relevance for all publications.

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
│   ├── raw/              # Raw Scopus data (not included in repository)
│   ├── interim/          # Abbreviation dictionaries, intermediate outputs
│   └── processed/        # Tokenized text, TF-IDF features, predictions
├── models/
│   ├── model_output/     # Evaluation metrics, confusion matrix, ROC curve
│   └── *.pkl             # Saved trained XGBoost model(s)
└── Makefile              # Optional automated pipeline runner
```

---

# 📦 Data Description

⚠️ Important data-sharing notice
The raw publication data (titles, abstracts, metadata) were retrieved from Elsevier’s Scopus API.
Under Scopus data-sharing conditions, full abstracts, titles, and metadata cannot be redistributed publicly.
Therefore:
data/raw/ is NOT included in this repository
Only processed tokens, abbreviation dictionaries, and model outputs generated from the Scopus data are included
Users who want to reproduce the workflow must retrieve the publication metadata themselves using Scopus and the provided EIDs (if you choose to release them)
data/raw/
This directory must be created by the user.
To fully reproduce the pipeline, populate data/raw/ with:
Pubs_labeled.csv
Human-annotated publications with binary labels
(1 = child-health relevant, 0 = not relevant)
(Labels can be shared — but Scopus metadata cannot)
Pubs_df.csv
Publication metadata (title, abstract, journal, subject areas) retrieved by the user from Scopus
data/interim/
(Automatically generated — allowed to share)
abbreviation_dicts_abs.json
updated_abbreviation_dicts_abs.json
These contain abbreviation mappings derived from processed text, and do not contain Scopus-restricted metadata.
data/processed/
(Safe to share — contains only cleaned tokens and predictions)
Pubs_processed_tokens.csv
Tokenized text derived from abstracts
(No raw abstract text is included)
Pubs_with_predictions.csv
Final classifier outputs
models/
Contains:
child_health_xgb_pipeline.pkl
Trained TF-IDF + XGBoost pipeline
Evaluation metrics (confusion matrix, ROC, etc.)
---

# 🔧 Pipeline Components

The pipeline has **four main steps**, each implemented as a Python script.

---

## **1. Build abbreviation dictionaries**  
**`scripts/make_abbreviation_dicts.py`**

This script:

- Detects abbreviations in abstracts  
- Saves JSON dictionaries into `data/interim/`

Output files:

- `abbreviation_dicts_abs.json`  
- `updated_abbreviation_dicts_abs.json`

---

## **2. Clean and tokenize text**  
**`scripts/make_processed_text.py`**

This step:

- Expands abbreviations  
- Removes noise and normalizes text  
- Tokenizes titles and abstracts  
- Generates `text` and `title_join` fields for TF-IDF  
- Outputs the final processed token CSV

Output:

- `data/processed/Pubs_processed_tokens.csv`

---

## **3. Train the classifier**

### **A. Default model training**
**`scripts/train_model.py`**

Performs:

- TF-IDF vectorization  
- XGBoost training  
- Train/test split  
- Model evaluation  
- Saves trained model + metrics

### **B. Hyperparameter optimization (optional)**
**`scripts/train_model_hyperparam.py`**

- Uses **RandomizedSearchCV**  
- Tests 100 configurations  
- Saves the best-performing model

Output:

- `models/child_health_xgb_pipeline.pkl`  
- Evaluation results in `models/model_output/`

---

## **4. Predict relevance on the full corpus**  
**`scripts/predict_labels.py`**

This script:

- Loads the trained model  
- Applies TF-IDF pipeline  
- Predicts relevance for *all* publications  
- Saves results as a new CSV

Output:

- `data/processed/Pubs_with_predictions.csv`

---

# 🚀 Running the Pipeline

There are two ways to run the workflow.

---

## **Option 1 — Using the Makefile (recommended)**

From the repository root:

```bash
make
```

This executes:

1. `make_abbreviation_dicts.py`
2. `make_processed_text.py`
3. `train_model.py`
4. `predict_labels.py`

You can also run steps individually:

```bash
make abbrev
make process
make train
make train_hyper
make predict
```

---

## **Option 2 — Run scripts manually**

### 1. Build abbreviation dictionaries

```bash
python scripts/make_abbreviation_dicts.py     --input_csv data/raw/Pubs_df.csv     --out_original_json data/interim/abbreviation_dicts_abs.json     --out_updated_json data/interim/updated_abbreviation_dicts_abs.json
```

### 2. Process text

```bash
python scripts/make_processed_text.py     --input_csv data/raw/Pubs_df.csv     --abbrev_json data/interim/updated_abbreviation_dicts_abs.json     --output_csv data/processed/Pubs_processed_tokens.csv
```

### 3. Train default model

```bash
python scripts/train_model.py     --labels_csv data/raw/Pubs_labeled.csv     --features_csv data/processed/Pubs_processed_tokens.csv     --output_model models/child_health_xgb_pipeline.pkl     --metrics_dir models/model_output
```

### 3b. Train with hyperparameter tuning

```bash
python scripts/train_model_hyperparam.py     --labels_csv data/raw/Pubs_labeled.csv     --features_csv data/processed/Pubs_processed_tokens.csv     --output_model models/child_health_xgb_pipeline.pkl     --metrics_dir models/model_output
```

### 4. Predict relevance

```bash
python scripts/predict_labels.py     --features_csv data/processed/Pubs_processed_tokens.csv     --model_path models/child_health_xgb_pipeline.pkl     --output_csv data/processed/Pubs_with_predictions.csv
```

---

# 📊 Model Description

**Model:** XGBoost (binary classifier)  
**Features:** TF-IDF vectors of combined title + abstract  
**Labels:** Human-labeled training dataset  
**Evaluation:**  
- Accuracy  
- Precision  
- Recall  
- F1 score  
- Confusion matrix  
- Classification report  
---



# 📬 Contact

**Masoumeh Dehghani**  
Data Analyst, Alberta Children’s Hospital Research Institute (ACHRI) , University of Calgary
Email: *[masoumeh.dehghanimog@ucalgary.ca , dm.dehghani@gmail.com]*  
