#!/usr/bin/env python
"""
Grid-search over NMF topic numbers and compute:
- reconstruction error
- c_v coherence (gensim)
- silhouette score on document-topic matrix W

Usage:
    python scripts/find_optimal_nmf_topics.py \
        --input_csv data/processed/CHW_pubs_topic_tokens.csv \
        --token_col filtered_3_token_processed_abstracts_title_ngrams \
        --metrics_csv data/processed/nmf_topic_search_results.csv \
        --fig_dir figures/topic_model \
        --min_topics 5 \
        --max_topics 200 \
        --step 5
"""

import argparse
import ast
import os
from typing import Any, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def safe_eval_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(t) for t in x]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return [str(t) for t in val]
            except Exception:
                return [s]
        return [s]
    return [str(x)]


def tokens_to_corpus(tokens_series: pd.Series) -> (List[str], List[List[str]]):
    """
    Convert a Series of token lists into:
      - corpus: list of strings with underscore-joined multiword tokens
      - texts: list of lists of tokens for gensim
    """
    processed_docs = []
    texts = []
    for tokens in tokens_series:
        toks = safe_eval_list(tokens)
        # join multiword tokens with underscore if they contain spaces
        joined = ["_".join(str(t).split()) for t in toks]
        processed_docs.append(" ".join(joined))
        texts.append(joined)
    return processed_docs, texts


def extract_topics(model: NMF, vectorizer: TfidfVectorizer, n_top_words: int = 10):
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for comp in model.components_:
        top_idx = comp.argsort()[:-n_top_words - 1:-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def find_optimal_topics(
    input_csv: str,
    token_col: str,
    metrics_csv: str,
    fig_dir: str,
    min_topics: int,
    max_topics: int,
    step: int,
    random_state: int = 42,
):
    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    if token_col not in df.columns:
        raise ValueError(f"Column '{token_col}' not found in input CSV.")

    corpus, texts = tokens_to_corpus(df[token_col])
    dictionary = Dictionary(texts)

    tfidf_vectorizer = TfidfVectorizer()
    dtm_nmf = tfidf_vectorizer.fit_transform(corpus)

    results = []

    for n_topics in range(min_topics, max_topics + 1, step):
        print(f"Evaluating NMF with {n_topics} topics...")
        nmf_model = NMF(
            n_components=n_topics, max_iter=1000, random_state=random_state
        )
        W = nmf_model.fit_transform(dtm_nmf)
        H = nmf_model.components_
        reconstruction_error = nmf_model.reconstruction_err_

        # Topics & coherence
        topic_words = extract_topics(nmf_model, tfidf_vectorizer, n_top_words=10)
        coherence_model = CoherenceModel(
            topics=topic_words, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_score = coherence_model.get_coherence()

        # Silhouette score on document-topic weights
        try:
            labels = W.argmax(axis=1)
            silhouette_avg = silhouette_score(W, labels)
        except Exception:
            silhouette_avg = np.nan

        results.append(
            {
                "n_topics": n_topics,
                "reconstruction_error": reconstruction_error,
                "coherence_score": coherence_score,
                "silhouette_score": silhouette_avg,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(metrics_csv, index=False)
    print(f"Saved NMF topic search metrics to: {metrics_csv}")

    # Plots
    matplotlib.rcParams["svg.fonttype"] = "none"

    # Reconstruction error
    plt.figure(figsize=(7, 4))
    plt.plot(
        results_df["n_topics"],
        results_df["reconstruction_error"],
        marker="o",
        linestyle=(0, (3, 1)),
        color="black",
    )
    plt.xlabel("Number of Topics")
    plt.ylabel("Reconstruction Error")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "nmf_reconstruction_error_vs_topics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(fig_dir, "nmf_reconstruction_error_vs_topics.svg"),
        format="svg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Coherence
    plt.figure(figsize=(7, 4))
    plt.plot(
        results_df["n_topics"],
        results_df["coherence_score"],
        marker="o",
        linestyle=(0, (3, 1)),
        color="black",
    )
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "nmf_coherence_vs_topics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(fig_dir, "nmf_coherence_vs_topics.svg"),
        format="svg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Find optimal number of NMF topics.")
    parser.add_argument("--input_csv", required=True, help="Processed input CSV.")
    parser.add_argument(
        "--token_col",
        default="filtered_3_token_processed_abstracts_title_ngrams",
        help="Column name with final token lists.",
    )
    parser.add_argument(
        "--metrics_csv",
        required=True,
        help="Path to save metrics CSV for topic search.",
    )
    parser.add_argument(
        "--fig_dir",
        default="figures/topic_model",
        help="Directory to save evaluation plots.",
    )
    parser.add_argument("--min_topics", type=int, default=5)
    parser.add_argument("--max_topics", type=int, default=200)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    find_optimal_topics(
        input_csv=args.input_csv,
        token_col=args.token_col,
        metrics_csv=args.metrics_csv,
        fig_dir=args.fig_dir,
        min_topics=args.min_topics,
        max_topics=args.max_topics,
        step=args.step,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
