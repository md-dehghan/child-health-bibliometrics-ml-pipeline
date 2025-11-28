#!/usr/bin/env python
"""
Train a child-health relevance classifier using TF-IDF features and XGBoost
with hyperparameter tuning via RandomizedSearchCV.

Usage:
    python scripts/train_model_hyperparam.py \
        --labels_csv data/raw/Pubs_labeled.csv \
        --features_csv data/processed/Pubs_processed_tokens.csv \
        --output_model models/child_health_xgb_pipeline_tuned.pkl \
        --test_size 0.2 \
        --random_state 42 \
        --metrics_dir models/model_output \
        --n_iter 100 \
        --cv 5 \
        --scoring f1
"""

import argparse
import ast
import os
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
# Main training + hyperparameter tuning logic
# ---------------------------------------------------------------------


def train_model_with_hyperparam(
    labels_csv: str,
    features_csv: str,
    output_model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    metrics_dir: str = "data/model_output",
    n_iter: int = 100,
    cv: int = 5,
    scoring: str = "f1",
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

    print("Training columns:")
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

    # 8. Base XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # 9. Full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb_clf),
        ]
    )

    # 10. Hyperparameter search space (same as notebook)
    param_distributions = {
        "classifier__n_estimators": [100,150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
        "classifier__max_depth": [3, 4, 5, 6, 7, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "classifier__subsample": [0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    print("Starting RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    # 11. Fit hyperparameter search on training data
    random_search.fit(X_train, y_train)

    print(f"\nBest parameters found: {random_search.best_params_}")
    print(f"Best CV {scoring} score: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_

    # 12. Evaluate best model on the held-out test set
    print("\nEvaluating best model on test set...")
    y_pred = best_model.predict(X_test)
    y_proba = (
        best_model.predict_proba(X_test)[:, 1]
        if hasattr(best_model, "predict_proba")
        else None
    )

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    conf_matrix = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=4)

    print("\n=== Test Metrics (best model) ===")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(cls_report)

    # 13. Save metrics, best params, and confusion matrix
    os.makedirs(metrics_dir, exist_ok=True)

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
            "best_cv_score": [random_search.best_score_],
            "scoring": [scoring],
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

    # Best params as JSON
    best_params_path = os.path.join(metrics_dir, f"{base_name}_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(random_search.best_params_, f, indent=2)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {conf_matrix_path}")
    print(f"Saved classification report to: {report_path}")
    print(f"Saved best parameters to: {best_params_path}")

    # 14. Save best model pipeline
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(best_model, output_model)
    print(f"\nSaved tuned model pipeline to: {output_model}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + XGBoost relevance model with hyperparameter tuning."
    )
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
        default="models/child_health_xgb_pipeline_tuned.pkl",
        help="Path to save the tuned model pipeline (joblib).",
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
        help="Directory to save evaluation outputs (metrics, confusion matrix, best params).",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=100,
        help="Number of parameter settings sampled in RandomizedSearchCV (default: 100).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="f1",
        help="Scoring metric for hyperparameter search (default: 'f1').",
    )

    args = parser.parse_args()

    train_model_with_hyperparam(
        labels_csv=args.labels_csv,
        features_csv=args.features_csv,
        output_model=args.output_model,
        test_size=args.test_size,
        random_state=args.random_state,
        metrics_dir=args.metrics_dir,
        n_iter=args.n_iter,
        cv=args.cv,
        scoring=args.scoring,
    )


if __name__ == "__main__":
    main()
