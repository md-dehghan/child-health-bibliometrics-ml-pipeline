📘 Child-Health Relevance Classification Pipeline
Machine-learning workflow for classifying bibliometric publications as child-health relevant or not relevant
This repository contains the full computational pipeline used to classify a large corpus of bibliometric publications into child-health relevant vs. not relevant, using manual annotations, text preprocessing, TF-IDF vectorization, and supervised machine learning (XGBoost).
🧭 Overview
The workflow:
Processes bibliometric text (titles, abstracts, metadata)
Builds abbreviation dictionaries
Cleans and tokenizes text
Trains a supervised classifier using human-labeled examples
Predicts relevance for the full publication corpus
All steps are scripted and reproducible.
📂 Project Structure
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
📢 Data Availability & Scopus Licensing Restrictions
⚠️ Important notice
The raw publication metadata used in this pipeline (titles, abstracts, journals, subject areas) were retrieved from Elsevier’s Scopus API.
Under Scopus data-sharing conditions, researchers cannot redistribute:
full abstracts
full titles
journal metadata
subject areas
Scopus-derived identifiers (except EIDs, which may be shared)
Therefore, this repository does not include any files containing Scopus-restricted metadata.
What cannot be shared
❌ data/raw/Pubs_df.csv
❌ data/raw/Pubs_labeled.csv (if it includes titles/abstracts)
❌ Any raw metadata from Scopus
What can be shared
✔️ Abbreviation dictionaries
✔️ Tokenized text (non-reversible)
✔️ Predictions
✔️ Scripts and models
✔️ EIDs (optional)
If you wish to fully reproduce the workflow, you must retrieve the raw metadata yourself using Scopus and the provided EIDs (if included).
📦 Data Description
data/raw/
(Not included — must be created by the user)
This directory should contain two Scopus-derived CSVs:
Pubs_labeled.csv
Human-annotated publications with binary relevance labels
(1 = child-health relevant, 0 = not relevant)
(Only labels may be shared — but not the metadata)
Pubs_df.csv
Publication metadata (title, abstract, journal, etc.)
Must be retrieved by the user via Scopus.
data/interim/
(Generated automatically — safe to share)
abbreviation_dicts_abs.json
updated_abbreviation_dicts_abs.json
Stores abbreviation detection output and cleaned mappings.
data/processed/
(Generated automatically — safe to share)
Pubs_processed_tokens.csv
Tokenized and normalized text (no raw abstracts)
Pubs_with_predictions.csv
Classifier predictions + probabilities
These files contain irreversible tokens, not original text.
models/
(Generated after running training scripts — not included by default)
This directory is created when you run the model training scripts.
It contains:
child_health_xgb_pipeline.pkl
TF-IDF + XGBoost trained pipeline
model_output/
Evaluation results:
ROC curve
Confusion matrix
Precision/Recall/F1
Classification report
🔧 Pipeline Components
1. Abbreviation dictionary construction
scripts/make_abbreviation_dicts.py
Detects abbreviations in abstracts
Builds and saves JSON dictionaries
Outputs:
abbreviation_dicts_abs.json
updated_abbreviation_dicts_abs.json
2. Text processing & tokenization
scripts/make_processed_text.py
Expands abbreviations
Normalizes text
Tokenizes titles & abstracts
Generates fields used for TF-IDF
Output:
data/processed/Pubs_processed_tokens.csv
3. Train the classifier
A. Default model
scripts/train_model.py
Performs:
TF-IDF vectorization
Train/test split
XGBoost training
Metrics generation
B. Hyperparameter tuning (optional)
scripts/train_model_hyperparam.py
RandomizedSearchCV (100 sampled configurations)
Saves best-performing model
Outputs:
Trained model .pkl
Metrics in models/model_output/
4. Predict relevance
scripts/predict_labels.py
Loads trained model
Predicts relevance for full corpus
Saves predictions as CSV
Output:
data/processed/Pubs_with_predictions.csv
🚀 Running the Pipeline
Option 1 — Use the Makefile (recommended)
make
Runs all steps in sequence.
Run individual components:
make abbrev
make process
make train
make train_hyper
make predict
Option 2 — Run scripts manually
All example commands are included in the README (unchanged for brevity).
📊 Model Description
Model: XGBoost (binary classifier)
Features: TF-IDF vectors from processed title + abstract tokens
Labels: Human-annotated relevance labels
Metrics:
Accuracy
Precision
Recall
F1
ROC curve
Confusion matrix
📬 Contact
Masoumeh Dehghani
Data Analyst, Alberta Children’s Hospital Research Institute (ACHRI)
University of Calgary
📧 masoumeh.dehghanimog@ucalgary.ca | dm.dehghani@gmail.com
