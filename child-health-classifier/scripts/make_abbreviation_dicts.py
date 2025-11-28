#!/usr/bin/env python
"""
Build abbreviation dictionaries from publication abstracts.

Usage:
    python scripts/make_abbreviation_dicts.py \
        --input_csv data/raw/Pubs_df.csv \
        --out_original_json data/interim/abbreviation_dicts_abs.json \
        --out_updated_json data/interim/updated_abbreviation_dicts_abs.json
"""

import argparse
import json
import re
from collections import defaultdict
from typing import List, Dict, Any

import pandas as pd
import wordninja
from fuzzywuzzy import process  # pip install fuzzywuzzy


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def apply_wordninja_to_text(text: str) -> str:
    """
    Apply Wordninja to split concatenated words in raw text.

    Args:
        text (str): Raw text.

    Returns:
        str: Text with long concatenated tokens split.
    """
    if text is None:
        return ""

    text = str(text)
    tokens = text.split()

    processed_tokens = []
    for token in tokens:
        if len(token) > 20:  # threshold can be adjusted
            processed_tokens.extend(wordninja.split(token))
        else:
            processed_tokens.append(token)

    return " ".join(processed_tokens)


def preprocess_text(text: str) -> str:
    """
    Simple regex-based cleaning:
    - split hyphenated words
    - split camelCase
    - split slashes between words
    - remove possessive 's
    """
    if text is None:
        return ""
    text = str(text)

    text = re.sub(r'(\w)-(\w)', r'\1 \2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\w)/(\w)', r'\1 \2', text)
    text = re.sub(r"(\w+)'s\b", r'\1', text)
    return text


def split_capital_words(full_form: str) -> List[str]:
    """
    Keep as in your notebook: currently just splits on spaces.
    """
    words = full_form.strip().split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
    return expanded_words


def merge_consecutive_single_letters(words: List[str]) -> List[str]:
    """
    Merge consecutive single-letter tokens into one string.
    """
    merged_words = []
    buffer = []
    for word in words:
        if len(word) == 1:
            buffer.append(word)
        else:
            if buffer:
                merged_words.append("".join(buffer))
                buffer = []
            merged_words.append(word)
    if buffer:
        merged_words.append("".join(buffer))
    return merged_words


# ---------------------------------------------------------------------
# Abbreviation extraction per abstract
# ---------------------------------------------------------------------

def process_abbreviations_final_corrected(corpus: pd.Series) -> List[List[Dict[str, Any]]]:
    """
    Process abbreviations with corrected matching logic to handle combined abbreviations.

    Args:
        corpus (pd.Series): Text data (e.g., ninja-processed abstracts).

    Returns:
        list of list of dict: For each text, a list of details dicts.
    """
    stop_words = {'and', 'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with'}
    results = []

    for text in corpus.fillna("").tolist():
        text = preprocess_text(text)

        # full form (with possible spaces/hyphens) + abbreviation (upper case or digits)
        matches = re.findall(r"([\w\s\-']+)\s?\(([A-Z0-9]+)\)", text)

        text_results = []
        abbreviation_lookup = {}  # To store abbreviations and their full forms

        for full_form, abbrev in matches:
            abbrev_length = len(abbrev)

            # Skip single-letter abbreviations
            if abbrev_length == 1:
                continue

            # Replace previous abbreviations in the full form
            words = full_form.split()
            updated_full_form = " ".join(
                abbreviation_lookup.get(word, word) for word in words
            )

            # Remove stop words
            full_form_words = [
                word for word in split_capital_words(updated_full_form)
                if word.lower() not in stop_words
            ]
            matched_candidates = []

            # Forward matching logic
            for start_idx in range(len(full_form_words)):
                abbrev_idx = 0
                temp_matched_words = []
                for i in range(start_idx, len(full_form_words)):
                    word = full_form_words[i]
                    if abbrev_idx >= abbrev_length:
                        break
                    if word[0].lower() == abbrev[abbrev_idx].lower():
                        temp_matched_words.append((i, word))
                        abbrev_idx += 1
                    else:
                        break
                if abbrev_idx == abbrev_length:
                    matched_candidates.append(temp_matched_words)

            matched_words = []
            if matched_candidates:
                # Choose the most compact match
                matched_candidates.sort(
                    key=lambda x: (
                        max(idx for idx, _ in x) - min(idx for idx, _ in x),
                        -max(idx for idx, _ in x),
                    )
                )
                best_match = matched_candidates[0]
                matched_words = [word for _, word in best_match]
            else:
                # Fallback 1: 3 consecutive letter matches
                if abbrev_length > 3:
                    abbrev_chars = list(abbrev.lower())
                    for i in range(len(full_form_words) - 2):
                        match_count = 0
                        temp_matched_words = []
                        for j in range(3):
                            if (
                                i + j < len(full_form_words)
                                and full_form_words[i + j][0].lower() == abbrev_chars[match_count]
                            ):
                                temp_matched_words.append(full_form_words[i + j])
                                match_count += 1
                            else:
                                break
                        if match_count == 3:
                            matched_words = full_form_words[i:]
                            break

                # Fallback 2: first-letter matching
                if not matched_words:
                    first_letter_matches = [
                        (i, word) for i, word in enumerate(full_form_words)
                        if word[0].lower() == abbrev[0].lower()
                    ]
                    if len(first_letter_matches) > 1:
                        last_match_idx = first_letter_matches[-1][0]
                        matched_words = full_form_words[last_match_idx:]
                    elif len(first_letter_matches) == 1:
                        last_match_idx = first_letter_matches[0][0]
                        matched_words = full_form_words[last_match_idx:]

            valid = bool(matched_words)
            matched_words = merge_consecutive_single_letters(matched_words)
            cleaned_full_form = " ".join(matched_words) if matched_words else None

            if cleaned_full_form:
                abbreviation_lookup[abbrev] = cleaned_full_form

            text_results.append({
                "abbreviation": abbrev,
                "abbreviation_length": abbrev_length,
                "original_full_form": full_form,
                "cleaned_full_form": cleaned_full_form,
                "matched": valid,
            })

        results.append(text_results)

    return results


def build_abbreviation_dicts(abbreviation_results_final: List[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Convert per-abstract results into list of dicts: abbrev -> cleaned_full_form.
    """
    abbreviation_dicts_abs: List[Dict[str, str]] = []

    for abstract_results in abbreviation_results_final:
        abstract_dict = {
            result["abbreviation"]: result["cleaned_full_form"]
            for result in abstract_results
            if result["cleaned_full_form"]
        }
        abbreviation_dicts_abs.append(abstract_dict)

    return abbreviation_dicts_abs


# ---------------------------------------------------------------------
# Find repeated keys and group similar full terms
# ---------------------------------------------------------------------

def filter_repeated_keys_with_same_terms(repeated_keys):
    """
    Keep only keys that appear with more than one unique full term.
    """
    filtered_keys = {}
    for key, details in repeated_keys.items():
        unique_full_terms = set(full_term.lower() for _, full_term in details)
        if len(unique_full_terms) > 1:
            filtered_keys[key] = details
    return filtered_keys


def find_repeated_keys(abstracts_list, min_occurrences=2):
    """
    Find abbreviation keys that appear in at least min_occurrences abstracts.
    """
    key_occurrences = defaultdict(list)

    for idx, terms in enumerate(abstracts_list):
        for key, full_term in terms.items():
            key_occurrences[key].append((f"Abstract {idx + 1}", full_term))

    repeated_keys = {
        key: details
        for key, details in key_occurrences.items()
        if len(details) >= min_occurrences
    }

    return repeated_keys


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def group_full_terms_for_key(full_terms, threshold=85):
    """
    Group similar full terms for a single abbreviation using fuzzy matching.
    """
    grouped_terms = defaultdict(list)
    processed_terms = {}

    for term in full_terms:
        normalized_term = normalize_text(term)

        if processed_terms:
            match, score = process.extractOne(normalized_term, processed_terms.keys())
        else:
            match, score = (None, 0)

        if score >= threshold and match is not None:
            grouped_terms[processed_terms[match]].append(term)
        else:
            grouped_terms[normalized_term].append(term)
            processed_terms[normalized_term] = normalized_term

    return grouped_terms


def categorize_full_terms(filtered_repeated_keys, threshold=85):
    """
    For each abbreviation key, group its various full forms into clusters.
    """
    categorized_keys = {}
    for key, details in filtered_repeated_keys.items():
        full_terms = [full_term for _, full_term in details]
        grouped_terms = group_full_terms_for_key(full_terms, threshold)
        categorized_keys[key] = grouped_terms
    return categorized_keys


def replace_full_terms(abbreviation_dicts_abs, categorized_keys):
    """
    Replace each full term with its canonical group representative.
    """
    import copy

    abbreviation_dicts_abs_s = copy.deepcopy(abbreviation_dicts_abs)

    for abstract_dict in abbreviation_dicts_abs_s:
        for key, full_term in list(abstract_dict.items()):
            normalized_term = normalize_text(full_term)

            if key in categorized_keys:
                for canonical_term, variants in categorized_keys[key].items():
                    normalized_variants = [normalize_text(v) for v in variants]
                    if normalized_term in normalized_variants:
                        abstract_dict[key] = canonical_term
                        break

    return abbreviation_dicts_abs_s


def remove_invalid_for_key(
    abbreviation_dicts_abs: List[Dict[str, str]],
    key_to_check: str,
    valid_full_terms: List[str],
    keep_if_in_list: bool = True,
) -> List[Dict[str, str]]:
    """
    Remove the specified key from dictionaries based on a list of valid terms.

    If keep_if_in_list=True:
        keep entries where full_term is in valid list, remove others.
    If keep_if_in_list=False:
        remove entries where full_term is in valid list, keep others.
    """
    valid_full_terms_norm = [normalize_text(term) for term in valid_full_terms]

    for abstract_dict in abbreviation_dicts_abs:
        if key_to_check in abstract_dict:
            full_term_norm = normalize_text(abstract_dict[key_to_check])

            if keep_if_in_list:
                # remove if NOT in valid list
                if full_term_norm not in valid_full_terms_norm:
                    del abstract_dict[key_to_check]
            else:
                # remove if IS in valid list
                if full_term_norm in valid_full_terms_norm:
                    del abstract_dict[key_to_check]

    return abbreviation_dicts_abs


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build abbreviation dictionaries from abstracts.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with abstracts & titles.")
    parser.add_argument(
        "--out_original_json",
        default="data/interim/abbreviation_dicts_abs.json",
        help="Output path for original abbreviation dictionaries.",
    )
    parser.add_argument(
        "--out_updated_json",
        default="data/interim/updated_abbreviation_dicts_abs.json",
        help="Output path for updated (canonical) abbreviation dictionaries.",
    )
    args = parser.parse_args()

    # 1. Load data
    df = pd.read_csv(args.input_csv)
    if "abstracts" not in df.columns:
        raise ValueError("Input CSV must contain an 'abstracts' column.")

    # 2. Wordninja on abstracts
    df["ninja_processed_abstracts"] = df["abstracts"].fillna("").apply(apply_wordninja_to_text)

    # 3. Extract abbreviation candidates
    abbreviation_results_final = process_abbreviations_final_corrected(df["ninja_processed_abstracts"])
    abbreviation_dicts_abs = build_abbreviation_dicts(abbreviation_results_final)

    # 4. Find repeated keys and group full forms
    repeated_keys = find_repeated_keys(abbreviation_dicts_abs, min_occurrences=2)
    filtered_repeated_keys = filter_repeated_keys_with_same_terms(repeated_keys)
    categorized_keys_1 = categorize_full_terms(filtered_repeated_keys, threshold=85)

    # 5. Canonical replacement
    updated_abbreviation_dicts_abs = replace_full_terms(abbreviation_dicts_abs, categorized_keys_1)

    # 6. Manual cleaning for specific keys ('OH' and 'SD')

    # For 'OH': keep only these allowed full terms
    valid_oh_full_terms = [
        "oral health", "odds having", "orthostatic hypotension", "one health",
        "object hit", "ovarian hyperstimulation", "oat hulls", "offspring higher",
        "other healthcare", "overhydration", "out hospital",
    ]
    updated_abbreviation_dicts_abs = remove_invalid_for_key(
        updated_abbreviation_dicts_abs,
        key_to_check="OH",
        valid_full_terms=valid_oh_full_terms,
        keep_if_in_list=True,
    )

    # For 'SD': removed these
    sd_terms_to_remove = [
        "suddenly", "study groups comprised 161 participants mean",
        "statistically different", "ss cerebrovascular reactivity values mean",
        "shutdown", "sixty patients were enrolled tested mean", "stride variability",
        "schizophrenia", "standard chow",
        "selected continued medical therapy had mean", "showed accuracy",
        "stage i or ii breast cancer mean", "surgery age",
        "sample included 995 patients mean", "subcutaneous adipose tissue expression variance",
        "students pre intervention had mean", "scores increased between baseline mean",
        "stay population", "scale intelligence quotient", "spellers",
        "seen nonparticipating hospitals had mean",
        "survived undergo neurocognitive assessment mean", "significant ωq",
    ]
    updated_abbreviation_dicts_abs = remove_invalid_for_key(
        updated_abbreviation_dicts_abs,
        key_to_check="SD",
        valid_full_terms=sd_terms_to_remove,
        keep_if_in_list=False,
    )

    # 7. Save JSONs
    def save_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    save_json(abbreviation_dicts_abs, args.out_original_json)
    save_json(updated_abbreviation_dicts_abs, args.out_updated_json)

    print(f"Saved original dicts to: {args.out_original_json}")
    print(f"Saved updated dicts to : {args.out_updated_json}")


if __name__ == "__main__":
    main()
