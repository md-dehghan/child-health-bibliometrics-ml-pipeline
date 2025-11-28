#!/usr/bin/env python
"""
Use a trained TF-IDF + XGBoost classifier to predict child-health
relevance labels for a full dataset.

Usage:
    python scripts/predict_labels.py \
        --model_path models/child_health_xgb_pipeline.pkl \
        --features_csv data/processed/Pubs_processed_tokens.csv \
        --output_csv data/processed/Pubs_with_predicted_relevance.csv
"""

import argparse
import ast
import os
import joblib

import pandas as pd


# ---------------------------------------------------------------------
# Helpers (same as in train_model.py)
# ---------------------------------------------------------------------


def safe_eval_list(x):
    """
    Convert a string representation of a list back to a Python list
    if needed. If it's already a list, return as-is.
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
# Prediction logic
# ---------------------------------------------------------------------


def predict_labels(
    model_path: str,
    features_csv: str,
    output_csv: str,
):
    # 1. Load trained model pipeline
    print(f"Loading model from: {model_path}")
    pipeline = joblib.load(model_path)

    # 2. Load features data
    print(f"Loading features from: {features_csv}")
    df = pd.read_csv(features_csv)

    # Keep EID for output if present
    eid_series = df["EID"] if "EID" in df.columns else None

    # 3. Build 'text' and 'title_join' columns
    required_token_cols = [
        "token_processed_abstracts_ngrams_lower",
        "token_processed_title_ngrams_lower",
    ]
    missing = [c for c in required_token_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input features CSV is missing required token columns: {missing}")

    df["text"] = make_joined_text_column(df["token_processed_abstracts_ngrams_lower"])
    df["title_join"] = make_joined_text_column(df["token_processed_title_ngrams_lower"])

    # 4. Other columns used by your pipeline (fill if missing)
    if "journal" not in df.columns:
        df["journal"] = ""
    if "subjectAreas" not in df.columns:
        df["subjectAreas"] = ""

    # 5. Define X (features) with same drops as in training
    cols_to_drop = [
        "coverDate",
        "centers membership",
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
        "relevance",  # just in case it happens to be present
    ]

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Drop EID from X (pipeline was trained without EID)
    if "EID" in X.columns:
        X = X.drop(columns=["EID"])

    # 6. Run predictions
    print("Running predictions...")
    y_pred = pipeline.predict(X)

    # Predicted probability for the positive class, if available
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X)[:, 1]
    else:
        y_proba = None

    # 7. Build output dataframe
    out_df = df.copy()
    out_df["predicted_relevance"] = y_pred
    if y_proba is not None:
        out_df["predicted_relevance_proba"] = y_proba

    # Make sure EID is present in the output if we had it
    if eid_series is not None and "EID" not in out_df.columns:
        out_df["EID"] = eid_series

    # 8. Save predictions
    out_df = out_df[['EID','predicted_relevance','predicted_relevance_proba']]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to: {output_csv}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Predict relevance labels using a trained TF-IDF + XGBoost model.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to trained model pipeline (joblib/pkl).",
    )
    parser.add_argument(
        "--features_csv",
        required=True,
        help="Path to features CSV (EID + token/metadata columns).",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to save CSV with predicted labels.",
    )

    args = parser.parse_args()
    predict_labels(
        model_path=args.model_path,
        features_csv=args.features_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
