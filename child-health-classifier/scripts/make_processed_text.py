#!/usr/bin/env python
"""
Create processed tokens for title and abstract using abbreviation dictionaries.

Usage:
    python scripts/make_processed_text.py \
        --input_csv data/raw/Pubs_df.csv \
        --original_json data/interim/abbreviation_dicts_abs.json \
        --updated_json data/interim/updated_abbreviation_dicts_abs.json \
        --output_csv data/processed/Pubs_processed_tokens.csv
"""

import argparse
import json
import re
from collections import Counter
from typing import List, Dict, Any

import pandas as pd
import wordninja
import nltk
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy 
from typing import List


# If needed once per environment:
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")


# ---------------------------------------------------------------------
# Wordninja & basic helpers 
# ---------------------------------------------------------------------

def apply_wordninja_to_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    tokens = text.split()
    processed_tokens = []
    for token in tokens:
        if len(token) > 20:
            processed_tokens.extend(wordninja.split(token))
        else:
            processed_tokens.append(token)
    return " ".join(processed_tokens)


def preprocess_text_basic(text: str) -> str:
    """
    Same cleaning as before (hyphen, camelCase, slash, 's).
    """
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"(\w)-(\w)", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(\w)/(\w)", r"\1 \2", text)
    text = re.sub(r"(\w+)'s\b", r"\1", text)
    return text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


# ---------------------------------------------------------------------
# Stopwords (you merged NLTK + spaCy in notebook; here we use NLTK only
# to keep this script simple and dependency-light; you *can* add spaCy if desired.)
# ---------------------------------------------------------------------

nltk_stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
spacy_stop_words = set(nlp.Defaults.stop_words)
all_stop_words = spacy_stop_words | nltk_stop_words

# ---------------------------------------------------------------------
# POS helper for lemmatization
# ---------------------------------------------------------------------

def get_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# ---------------------------------------------------------------------
# Main abstract preprocessing using dictionaries
# ---------------------------------------------------------------------

def preprocess_abstract(
    abstract: str,
    original_abbreviation_dict: Dict[str, str],
    updated_abbreviation_dict: Dict[str, str],
) -> List[str]:
    """
    Your full pipeline:

    1. lowercase
    2. replace full terms with their keys (original dict)
    3. replace keys with placeholders (updated dict)
    4. clean punctuation
    5. tokenize
    6. remove stopwords + single letters
    7. POS-tag + lemmatize
    8. replace placeholders with full terms
    """
    placeholders = {}
    if abstract is None:
        abstract = ""
    abstract = abstract.lower()

    # Step 1: full term -> abbreviation key (from original dict)
    for key, full_term in original_abbreviation_dict.items():
        if full_term:
            pattern = rf"\b{re.escape(full_term.lower())}\b"
            abstract = re.sub(pattern, key.lower(), abstract)

    # Step 2: key -> placeholder (using updated dict)
    for key, full_term in updated_abbreviation_dict.items():
        placeholder = f"ABBR_{key}_PLACEHOLDER"
        pattern = rf"\b{key.lower()}\b"
        abstract = re.sub(pattern, placeholder, abstract)
        placeholders[placeholder] = full_term

    # Step 3: remove punctuation/special chars
    abstract_cleaned = re.sub(r"[^\w\s]", " ", abstract)

    # Step 4: tokenize
    tokens = word_tokenize(abstract_cleaned)

    # Step 5: remove stopwords
    tokens = [token for token in tokens if token.lower() not in all_stop_words]

    # Step 6: remove single-letter tokens
    tokens = [token for token in tokens if len(token) > 1]

    # Step 7: POS-tag + lemmatize
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(tag))
        for token, tag in pos_tags
    ]

    # Step 8: replace placeholders back to full terms
    tokens_with_full_terms = [
        placeholders.get(token, token) for token in tokens
    ]

    return tokens_with_full_terms


# ---------------------------------------------------------------------
# Token cleaning & n-grams
# ---------------------------------------------------------------------

def clean_tokens(tokens: List[str]) -> List[str]:
    """
    Remove tokens that are only integers or decimal numbers.
    """
    numeric_pattern = re.compile(r"^\d+(\.\d+)?$")
    return [t for t in tokens if not numeric_pattern.match(t)]


def generate_non_overlapping_frequent_ngrams_range(
    tokens_list: List[str],
    n_range=(3, 5),
    min_occurrences=3,
) -> Dict[int, List[str]]:
    """
    Generate non-overlapping n-grams for a range of n values,
    only on single-word tokens.
    """
    single_word_tokens = [token for token in tokens_list if " " not in token]

    frequent_ngrams: Dict[int, List[str]] = {}

    for n in range(n_range[0], n_range[1] + 1):
        ngram_list = [
            " ".join(single_word_tokens[i:i + n])
            for i in range(0, len(single_word_tokens), n)
            if len(single_word_tokens[i:i + n]) == n
        ]
        ngram_counts = Counter(ngram_list)
        frequent_ngrams[n] = [
            ngram for ngram, count in ngram_counts.items()
            if count >= min_occurrences
        ]

    return frequent_ngrams


def flatten_ngram_dict(ngram_dict: Dict[int, List[str]]) -> List[str]:
    return [ng for sub in ngram_dict.values() for ng in sub]



def replace_tokens_with_ngrams(tokens_list: List[str],
                               ngrams_list: List[str]) -> List[str]:
    """
    Replace sequences of tokens in the token list with equivalent n-grams,
    always preferring longer n-grams when there is overlap.
    """
    # Deduplicate and sort by token length (longest first)
    ngrams_sorted = sorted(
        set(ngrams_list),
        key=lambda x: len(x.split()),
        reverse=True
    )

    result_tokens = []
    skip_count = 0
    n = len(tokens_list)
    i = 0

    while i < n:
        if skip_count > 0:
            skip_count -= 1
            i += 1
            continue

        matched = False
        for ngram in ngrams_sorted:
            ngram_tokens = ngram.split()
            L = len(ngram_tokens)

            if tokens_list[i:i + L] == ngram_tokens:
                result_tokens.append(ngram)
                skip_count = L - 1
                matched = True
                break

        if not matched:
            result_tokens.append(tokens_list[i])

        i += 1

    return result_tokens


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create processed tokens for titles & abstracts.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with abstracts & titles.")
    parser.add_argument("--original_json", required=True, help="Path to original abbreviation_dicts_abs.json.")
    parser.add_argument("--updated_json", required=True, help="Path to updated_abbreviation_dicts_abs.json.")
    parser.add_argument(
        "--output_csv",
        default="data/processed/Pubs_processed_tokens.csv",
        help="Output CSV with processed token columns.",
    )
    args = parser.parse_args()

    # 1. Load data & dictionaries
    df = pd.read_csv(args.input_csv)

    if "abstracts" not in df.columns or "title" not in df.columns:
        raise ValueError("Input CSV must contain 'abstracts' and 'title' columns.")

    with open(args.original_json, "r", encoding="utf-8") as f:
        abbreviation_dicts_abs = json.load(f)

    with open(args.updated_json, "r", encoding="utf-8") as f:
        updated_abbreviation_dicts_abs = json.load(f)

    if len(abbreviation_dicts_abs) != len(df):
        print("Warning: length of abbreviation_dicts_abs != number of rows; "
              "will fall back to empty dicts where needed.")

    # 2. Wordninja on abstracts & titles
    df["ninja_processed_abstracts"] = df["abstracts"].fillna("").apply(apply_wordninja_to_text)
    df["ninja_processed_title"] = df["title"].fillna("").apply(apply_wordninja_to_text)

    # 3. Process abstracts and titles with dictionaries
    processed_abstracts: List[List[str]] = []
    processed_titles: List[List[str]] = []

    for idx, abstract in enumerate(df["ninja_processed_abstracts"]):
        original_dict = abbreviation_dicts_abs[idx] if idx < len(abbreviation_dicts_abs) else {}
        updated_dict = updated_abbreviation_dicts_abs[idx] if idx < len(updated_abbreviation_dicts_abs) else {}
        tokens = preprocess_abstract(abstract, original_dict, updated_dict)
        processed_abstracts.append(clean_tokens(tokens))

    for idx, title in enumerate(df["ninja_processed_title"]):
        original_dict = abbreviation_dicts_abs[idx] if idx < len(abbreviation_dicts_abs) else {}
        updated_dict = updated_abbreviation_dicts_abs[idx] if idx < len(updated_abbreviation_dicts_abs) else {}
        tokens = preprocess_abstract(title, original_dict, updated_dict)
        processed_titles.append(clean_tokens(tokens))

    df["token_processed_abstracts"] = processed_abstracts
    df["token_processed_title"] = processed_titles


    # 4. N-gram generation & replacement per row
    frequent_ngrams_list = []
    flat_ngrams_list = []
    abstracts_with_ngrams = []
    titles_with_ngrams = []

    for idx, row in df.iterrows():
        tokens_abs = row["token_processed_abstracts"]
        tokens_title = row["token_processed_title"]

        frequent_ngrams = generate_non_overlapping_frequent_ngrams_range(
            tokens_abs,
            n_range=(3, 5),
            min_occurrences=3,
        )
        flat_ngrams = flatten_ngram_dict(frequent_ngrams)

        tokens_abs_ng = replace_tokens_with_ngrams(tokens_abs, flat_ngrams)
        tokens_title_ng = replace_tokens_with_ngrams(tokens_title, flat_ngrams)

        # lower-case
        tokens_abs_ng = [t.lower() for t in tokens_abs_ng]
        tokens_title_ng = [t.lower() for t in tokens_title_ng]

        frequent_ngrams_list.append(frequent_ngrams)
        flat_ngrams_list.append(flat_ngrams)
        abstracts_with_ngrams.append(tokens_abs_ng)
        titles_with_ngrams.append(tokens_title_ng)

    df["frequent_ngrams"] = frequent_ngrams_list
    df["flat_frequent_ngrams"] = flat_ngrams_list
    df["token_processed_abstracts_ngrams_lower"] = abstracts_with_ngrams
    df["token_processed_title_ngrams_lower"] = titles_with_ngrams

    # 5. Save
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved processed data to: {args.output_csv}")


if __name__ == "__main__":
    main()
