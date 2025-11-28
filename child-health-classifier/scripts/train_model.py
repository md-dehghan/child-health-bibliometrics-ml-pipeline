#!/usr/bin/env python
"""
Train a child-health relevance classifier using TF-IDF features and XGBoost.

Usage:
    python scripts/train_model.py \
        --labels_csv data/raw/Pubs_labeled.csv \
        --features_csv data/processed/Pubs_processed_tokens.csv \
        --output_model models/child_health_xgb_pipeline.pkl \
        --test_size 0.2 \
        --random_state 42\
        --metrics_dir models/model_output
"""

import argparse
import ast
import os
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import xgboost as xgb


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def safe_eval_list(x):
    """
    Convert a string representation of a list back to a Python list
    if needed. If it's already a list, return as-is.

    Handles cases where the CSV saved list-like columns such as:
    "['child', 'health', 'research']"
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        # If it looks like a Python list, try literal_eval
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return val
            except Exception:
                # Fall back to space-splitting if literal_eval fails
                return s.split()
        # Otherwise treat as space-separated tokens
        return s.split()
    # Fallback
    return [str(x)]


def make_joined_text_column(tokens_series: pd.Series) -> pd.Series:
    """
    Take a Series of token lists (or list-like strings) and
    return a Series of space-separated strings with underscores
    for multi-word tokens:

        ' '.join('_'.join(token.split()) for token in x)
    """
    def join_tokens(x):
        tokens = safe_eval_list(x)
        return " ".join("_".join(str(tok).split()) for tok in tokens)

    return tokens_series.fillna("").apply(join_tokens)


# ---------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------


def train_model(
    labels_csv: str,
    features_csv: str,
    output_model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    metrics_dir: str = "data/model_output",

):
    # 1. Load data
    df_labels = pd.read_csv(labels_csv)
    df_features = pd.read_csv(features_csv)

    # Sanity checks
    if "EID" not in df_labels.columns or "EID" not in df_features.columns:
        raise ValueError("Both input CSVs must contain an 'EID' column for merging.")
    if "relevance" not in df_labels.columns:
        raise ValueError("The labels CSV must contain a 'relevance' column.")

    # 1a. Merge on EID:
    #     - df_features has tokens/metadata
    #     - df_labels has relevance
    df = pd.merge(df_features, df_labels[["EID", "relevance"]], on="EID", how="inner")

    print(f"Merged dataframe shape: {df.shape}")
    print(f"Number of unique EIDs after merge: {df['EID'].nunique()}")

    # 2. Required columns for building features
    required_cols = [
        "token_processed_abstracts_ngrams_lower",
        "token_processed_title_ngrams_lower",
        "relevance",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Merged data is missing required columns: {missing}")

    # 3. Build 'text' and 'title_join' columns
    df["text"] = make_joined_text_column(df["token_processed_abstracts_ngrams_lower"])
    df["title_join"] = make_joined_text_column(df["token_processed_title_ngrams_lower"])

    # Other columns used by your pipeline (fill if missing)
    if "journal" not in df.columns:
        df["journal"] = ""
    if "subjectAreas" not in df.columns:
        df["subjectAreas"] = ""

    # 4. Define X (features) and y (target)
    cols_to_drop = [
        "coverDate",
        "title",
        "lang",
        "abstracts",
        "subtypeDescription",
        "ninja_processed_abstracts",
        "ninja_processed_title",
        "token_processed_abstracts",
        "token_processed_title",
        "frequent_ngrams",
        "flat_frequent_ngrams",
        "token_processed_abstracts_ngrams",
        "token_processed_title_ngrams",
        "token_processed_abstracts_ngrams_lower",
        "token_processed_title_ngrams_lower",
        "relevance",
    ]

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    y = df["relevance"]

    # 5. Train/test split (on full X including EID; drop EID afterward)
    X_train_1, X_test_1, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Remove EID from features
    if "EID" in X_train_1.columns:
        X_train = X_train_1.drop(columns=["EID"])
        X_test = X_test_1.drop(columns=["EID"])
    else:
        X_train = X_train_1
        X_test = X_test_1

    print(X_train.columns)
    # 6. Define TF-IDF vectorizers
    tfidf_vectorizer_text = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_vectorizer_title = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_vectorizer_journal = TfidfVectorizer()

    # 7. ColumnTransformer to apply TF-IDF on text columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", tfidf_vectorizer_text, "text"),
            ("title_join", tfidf_vectorizer_title, "title_join"),
            ("journal", tfidf_vectorizer_journal, "journal"),
            ("subjectAreas", tfidf_vectorizer_journal, "subjectAreas"),
        ],
        remainder="passthrough",
    )

    # 8. XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        random_state=random_state
    )

    # 9. Full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb_clf),
        ]
    )

    # 10. Train
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 11. Evaluate
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    conf_matrix = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=4)

    print("\n=== Metrics ===")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(cls_report)

    # 12. Save metrics & confusion matrix
    os.makedirs(metrics_dir, exist_ok=True)

    # Use model file name as base
    base_name = os.path.splitext(os.path.basename(output_model))[0]

    # Metrics summary as CSV
    metrics_path = os.path.join(metrics_dir, f"{base_name}_metrics.csv")
    metrics_df = pd.DataFrame(
        {
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "roc_auc": [roc_auc],
        }
    )
    metrics_df.to_csv(metrics_path, index=False)

    # Confusion matrix as CSV
    conf_matrix_path = os.path.join(metrics_dir, f"{base_name}_confusion_matrix.csv")
    conf_df = pd.DataFrame(
        conf_matrix,
        columns=["pred_0", "pred_1"],
        index=["true_0", "true_1"],
    )
    conf_df.to_csv(conf_matrix_path)

    # Classification report as text
    report_path = os.path.join(metrics_dir, f"{base_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(cls_report)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {conf_matrix_path}")
    print(f"Saved classification report to: {report_path}")

    # 13. Save model
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(pipeline, output_model)
    print(f"\nSaved trained model pipeline to: {output_model}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + XGBoost relevance model.")
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="Path to labels CSV (EID + relevance).",
    )
    parser.add_argument(
        "--features_csv",
        required=True,
        help="Path to features CSV (EID + token/metadata columns).",
    )
    parser.add_argument(
        "--output_model",
        default="models/child_health_xgb_pipeline.pkl",
        help="Path to save the trained model pipeline (joblib).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--metrics_dir",
        default="data/model_output",
        help="Directory to save evaluation outputs (metrics, confusion matrix).",
    )

    args = parser.parse_args()
    train_model(
        labels_csv=args.labels_csv,
        features_csv=args.features_csv,
        output_model=args.output_model,
        test_size=args.test_size,
        random_state=args.random_state,
        metrics_dir=args.metrics_dir,

    )


if __name__ == "__main__":
    main()
