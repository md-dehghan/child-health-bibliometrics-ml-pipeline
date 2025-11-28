#!/usr/bin/env python
"""
Prepare token data for topic modeling.

- Loads a CSV with tokenized columns:
    * token_processed_abstracts_ngrams_lower
    * token_processed_title_ngrams_lower
- Combines title + abstract tokens
- Applies sequence-based replacements (theory_mind, quality_life, MIS)
- Applies one-to-one token replacements (replacements dict)
- Removes rare/common/extra tokens
- Saves a processed CSV with:
    * filtered_token_processed_abstracts_title_ngrams
    * filtered_2_token_processed_abstracts_title_ngrams
    * filtered_3_token_processed_abstracts_title_ngrams
- Saves pre- and post-filter token frequency plots.

Usage:
    python scripts/prepare_topic_model_data.py \
        --input_csv data/raw/CHW_pubs.csv \
        --output_csv data/processed/CHW_pubs_topic_tokens.csv \
        --fig_dir figures/topic_model
"""

import argparse
import ast
import os
from collections import Counter
from typing import List, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Constants: white lists and extra words
# ---------------------------------------------------------------------

WHITELIST_TOKENS = ['activation','active','activity','acute','anxiety','adolescent','adult','asthma','autism spectrum disorder','bone','brain','cancer',
'cardiac','congenital','covid','cognitive','growth','cost','infection','injury','pregnancy','pain','stress','depression',
'diabetes','diet','dna','disability','drug','education','emotional','epilepsy',
'exercise','family','healthcare','guideline','heart','impairment','income','infection','inflammatory','insulin','intervention',
'lung','metabolic','mitochondrial','mortality','morbidity','muscle','neonatal','pandemic','pregnancy','pregnant','prenatal',
'rna','school','safety','sex','sexual','sleep','society','social','status','stress','student','surgery','surgical','tumor',
'vaccine','university','variant',
'delivery','gene','genetic','genome','mortality','mutation','pain','sequence',
'behavior','behavioral','behaviour','caregiver','cellular','cell',
'child','childhood',
'community','culture','death','deficiency',
'development','developmental',
'environment','environmental',
'fetal','food','gestational','girl','grade','immune','infant','language','learn','lung','mass','maternal','mental',
'mother','motor','obesity',
'parent','parental',
'pediatric','physical','play','policy','preterm','psychological','receptor','relationship','relative',
'respiratory','son','stem','syndrome','variant','woman','young','youth','age','hokey']

EXTRA_WORDS = ['aor','march','woman','men','theory','participation', 'determinant','min','coverage','attachment','externalize','internalize','return','million','nmol',
               'ontario', 'million', 'anti', 'africa', 'novo','chinese','kenya','generalize','exclusive','inclusive','inclusion','cop','initiation','exclusively', 'exclusivity','chapter','facility','aid','perspective','restriction',
               'atypical','incremental','utilization','trajectory','leave','session','pilot', 'scoping', 'feasibility', 'indicator', 'theme', 'transfer','item','unit','intensive','mild','acute',
               'middle', 'uptake', 'illness','organize', 'extremely',
               'adolescent', 'adolescent_young_adult','young', 'youth', 'girl', 'adult', 'boy', 'member', 'age',
               'randomized_controlled_trial', 'project', 'participatory', 'presentation','framework', 'adulthood', 'child', 'classification', 'healthy_controls','trend','person',
               'healthy_controls',
               'axis', "randomized controlled trial", "adolescent young adult", "healthy controls",'candidate','aya','rating','implement','researcher','odds_ratio'
]
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def safe_eval_list(x: Any) -> List[Any]:
    """Convert string representation of a list back to a Python list if needed."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return val
            except Exception:
                return [s]
        return [s]
    return [x]


def ensure_list(x: Any) -> List[Any]:
    """Ensure the value is a list (no NaNs)."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return safe_eval_list(x)


def combine_tokens(row) -> List[str]:
    """
    Combine abstract and title ngram lists into one flat token list,
    filtering tokens with len <= 2.
    """
    abs_tokens = ensure_list(row["token_processed_abstracts_ngrams_lower"])
    title_tokens_raw = ensure_list(row["token_processed_title_ngrams_lower"])

    # Flatten nested lists in title tokens if present
    title_tokens: List[str] = []
    for item in title_tokens_raw:
        if isinstance(item, list):
            title_tokens.extend(item)
        else:
            title_tokens.append(item)

    combined = abs_tokens + title_tokens
    combined = [str(t) for t in combined if len(str(t)) > 2]
    return combined


# def apply_sequence_replacements(tokens: List[str]) -> List[str]:
#     """
#     Apply sequence-based replacements:
#     - theory mind / tom -> theory_mind
#     - quality life / qol -> quality_life
#     - inflammatory syndrome / MIS + multisystem inflammatory syndrome -> multisystem_inflammatory_syndrome
#     """
#     # 1) theory + mind
#     if any(tokens[i] == "theory" and tokens[i + 1] == "mind" for i in range(len(tokens) - 1)):
#         tokens = ["theory_mind" if t == "tom" else t for t in tokens]
#         # collapse sequence
#         merged = []
#         i = 0
#         while i < len(tokens):
#             if i < len(tokens) - 1 and tokens[i] == "theory" and tokens[i + 1] == "mind":
#                 merged.append("theory_mind")
#                 i += 2
#             else:
#                 merged.append(tokens[i])
#                 i += 1
#         tokens = merged

#     # 2) quality + life
#     if any(tokens[i] == "quality" and tokens[i + 1] == "life" for i in range(len(tokens) - 1)):
#         tokens = ["quality_life" if t == "qol" else t for t in tokens]
#         merged = []
#         i = 0
#         while i < len(tokens):
#             if i < len(tokens) - 1 and tokens[i] == "quality" and tokens[i + 1] == "life":
#                 merged.append("quality_life")
#                 i += 2
#             else:
#                 merged.append(tokens[i])
#                 i += 1
#         tokens = merged

#     # 3) MIS: inflammatory syndrome / multisystem inflammatory syndrome
#     if any(tokens[i] == "inflammatory" and tokens[i + 1] == "syndrome" for i in range(len(tokens) - 1)):
#         tokens = ["multisystem_inflammatory_syndrome" if t == "mis" else t for t in tokens]

#     merged = []
#     i = 0
#     while i < len(tokens):
#         if (
#             i < len(tokens) - 2
#             and tokens[i] == "multisystem"
#             and tokens[i + 1] == "inflammatory"
#             and tokens[i + 2] == "syndrome"
#         ):
#             merged.append("multisystem_inflammatory_syndrome")
#             i += 3
#         else:
#             merged.append(tokens[i])
#             i += 1

#     return merged
from typing import List

def apply_sequence_replacements(tokens: List[str]) -> List[str]:
    """
    Make this behave like the notebook version:

    - If there is at least one 'theory' 'mind' sequence:
        * replace ALL 'tom' -> 'theory_mind'
        * then merge ONLY THE FIRST 'theory' 'mind' pair into 'theory_mind'
    - If there is at least one 'quality' 'life' sequence:
        * replace ALL 'qol' -> 'quality_life'
        * then merge ONLY THE FIRST 'quality' 'life' pair into 'quality_life'
    - If there is at least one 'inflammatory' 'syndrome' sequence:
        * replace ALL 'mis' -> 'multisystem_inflammatory_syndrome'
        * then merge ONLY THE FIRST 'multisystem' 'inflammatory' 'syndrome'
          triple into 'multisystem_inflammatory_syndrome'
    """

    if not isinstance(tokens, list):
        return tokens
    tokens = list(tokens)

    # ------------------------------------------------------------------
    # 1) theory + mind / tom -> theory_mind  
    # ------------------------------------------------------------------
    if any(tokens[i] == "theory" and tokens[i + 1] == "mind"
           for i in range(len(tokens) - 1)):
        tokens = ["theory_mind" if t == "tom" else t for t in tokens]

        for i in range(len(tokens) - 1):
            if tokens[i] == "theory" and tokens[i + 1] == "mind":
                tokens = tokens[:i] + ["theory_mind"] + tokens[i+2:]
                break  

    # ------------------------------------------------------------------
    # 2) quality + life / qol -> quality_life
    # ------------------------------------------------------------------
    if any(tokens[i] == "quality" and tokens[i + 1] == "life"
           for i in range(len(tokens) - 1)):
        tokens = ["quality_life" if t == "qol" else t for t in tokens]

        for i in range(len(tokens) - 1):
            if tokens[i] == "quality" and tokens[i + 1] == "life":
                tokens = tokens[:i] + ["quality_life"] + tokens[i+2:]
                break  

    # ------------------------------------------------------------------
    # 3) MIS / multisystem inflammatory syndrome
    # ------------------------------------------------------------------
    if any(tokens[i] == "inflammatory" and tokens[i + 1] == "syndrome"
           for i in range(len(tokens) - 1)):
        # Replace all 'mis' with 'multisystem_inflammatory_syndrome'
        tokens = [
            "multisystem_inflammatory_syndrome" if t == "mis" else t
            for t in tokens
        ]

    for i in range(len(tokens) - 2):
        if (tokens[i] == "multisystem"
            and tokens[i + 1] == "inflammatory"
            and tokens[i + 2] == "syndrome"):
            tokens = (
                tokens[:i]
                + ["multisystem_inflammatory_syndrome"]
                + tokens[i+3:]
            )
            break 

    return tokens


def filter_tokens(tokens: List[str], tokens_to_remove: set) -> List[str]:
    return [t for t in tokens if t not in tokens_to_remove]


def plot_token_freq(token_lists: pd.Series, out_png: str, out_svg: str, title_suffix: str):
    """Plot Zipf-like token frequency distribution (log scale Y)."""
    all_tokens = [tok for doc in token_lists for tok in doc]
    token_counts = Counter(all_tokens)
    freqs = sorted(token_counts.values(), reverse=True)

    matplotlib.rcParams["svg.fonttype"] = "none"
    plt.figure(figsize=(6, 4))
    plt.plot(freqs, color="black", linewidth=1)
    plt.xlabel("Rank", fontsize=10)
    plt.ylabel("Frequency (Log Scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def prepare_topic_data(input_csv: str, output_csv: str, fig_dir: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Ensure list-like columns are parsed properly
    for col in [
        "token_processed_abstracts_ngrams_lower",
        "token_processed_title_ngrams_lower",
    ]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in input CSV.")
        df[col] = df[col].apply(ensure_list)

    # Combine abstract + title
    df["filtered_token_processed_abstracts_title_ngrams"] = df.apply(
        combine_tokens, axis=1
    )

    # # Plot initial frequency distribution
    # plot_token_freq(
    #     df["filtered_token_processed_abstracts_title_ngrams"],
    #     out_png=os.path.join(fig_dir, "token_freq_before_filter.png"),
    #     out_svg=os.path.join(fig_dir, "token_freq_before_filter.svg"),
    #     title_suffix="Before Filtering",
    # )

    # Compute rare and common tokens
    all_tokens = [
        tok for doc in df["filtered_token_processed_abstracts_title_ngrams"] for tok in doc
    ]
    token_counts = Counter(all_tokens)

    min_threshold = 2  # keep tokens with freq >= 2
    max_threshold = 0.05 * len(df)  # appear in at most 5% of documents

    rare_tokens = {t for t, c in token_counts.items() if c < min_threshold}
    common_tokens = {t for t, c in token_counts.items() if c > max_threshold}

    whitelist = set(WHITELIST_TOKENS)
    filtered_common_tokens = {t for t in common_tokens if t not in whitelist}
    tokens_to_remove = set(rare_tokens | filtered_common_tokens | set(EXTRA_WORDS))

    # Apply sequence replacements
    df["filtered_token_processed_abstracts_title_ngrams"] = df[
        "filtered_token_processed_abstracts_title_ngrams"
    ].apply(apply_sequence_replacements)

    # One-to-one replacements dictionary
    replacements = {
        "autism": "autism_spectrum_disorder",
        "autistic": "autism_spectrum_disorder",
        "cov": "covid",
        "coronavirus": "covid",
        "pregnant": "pregnancy",
        "mirnas": "mirna",
        "sars": "covid",
        "childhood": "child",
        "adhd": "attention_deficit_hyperactivity_disorder",
        "ed": "emergency_department",
        "behavioral": "behavior",
        "behaviour": "behavior",
        "cellular": "cell",
        "developmental": "development",
        "environmental": "environment",
        "parental": "parent",
        "gestational": "gestation",
        "homelessness": "homeless",
        "analgesia": "analgesic",
        "genetics": "gene",
        "genetic": "gene",
        "genomic": "genome",
        "educational": "education",
        "epileptic": "epilepsy",
        "ace": "adverse_childhood_experiences",
        "bmi": "body_mass_index",
        "sexual": "sex",
        "obese": "obesity",
        "sexually": "sex",
        "adolescents": "adolescent",
        "adolescence": "adolescent",
        "suicidal": "suicide",
        "diabetic": "diabete",
        "diabetes": "diabete",
        "depressive": "depress",
        "depression": "depress",
        "allergic": "allergy",
        "vaccinate": "vaccine",
        "vaccination": "vaccine",
        "reading": "read",
        "neonatal": "neonate",
        "perinatal": "prenatal",
        "hba1c": "glycated_hemoglobin",
        "ayas": "adolescent_young_adult",
        "mutant": "mutation",
        "surgical": "surgery",
        "hpv": "human_papillomaviruses",
        "asd": "autism_spectrum_disorder",
        "icf": "international_classification_functioning_disability_health",
        "reader": "read",
        "caregiving": "caregiver",
        "mri": "magnetic_resonance_imaging",
        "aggressive": "aggression",
        "cannabinoids": "cannabis",
        "suicidality": "suicide",
        "asthmatic": "asthma",
        "paediatric": "pediatric",
        "training": "train",
        "metabolic": "metabolism",
        "mitochondrion": "mitochondrial",
        "prescribe": "prescription",
        "breastfed": "breastfeed",
        "breastfeeding": "breastfeed",
        "neonates": "neonate",
        "placental": "placenta",
        "culturally": "culture",
        "cultural": "culture",
        "anxious": "anxiety",
        "heart": "cardiac",
        "stressor": "stress",
        "cftr": "cystic_fibrosis_transmembrane_regulator",
        "cf_transmembrane_regulator": "cystic_fibrosis_transmembrane_regulator",
        "cf transmembrane regulator": "cystic_fibrosis_transmembrane_regulator",
        "feeding": "feed",
        "inflammatory": "inflammation",
        "supplementation": "supplement",
        "underwent": "undergo",
        "emotional": "emotion",
        "sexuality": "sex",
        "mtbi": "mild_traumatic_brain_injury",
        "dietary": "diet",
        "parenting": "parent",
        "painful": "pain",
        "feeing": "feed",
        "cannabinoid": "cannabis",
        "cannabidiol": "cannabis",
        "engagement": "engage",
        "snp": "single_nucleotide_polymorphism",
        "bear": "born",
        "brca1": "breast_cancer",
        "die": "death",
        "depressed": "depress",
        "paternal": "father",
        "uninfect": "infection",
        "infect": "infection",
        "prenatally": "prenatal",
        "uninfected": "infection",
        "bipolar_disorder": "bipolar",
        "electroencephalography": "electroencephalograph_y",
        "eeg": "electroencephalograph_y",
        "traumatic": "trauma",
        "mild_traumatic_brain_injury": "traumatic_brain_injury",
        "sedentary_behaviour": "sedentary",
        "drinking": "drink",
        "nutritional": "nutrition",
        "disabilities": "disability",
        "bipolar disorder": "bipolar",
        "adiposity": "adipose",
        "crohn": "crohn_disease",
    }

    # 2nd-level filtered tokens with replacements
    df["filtered_2_token_processed_abstracts_title_ngrams"] = df[
        "filtered_token_processed_abstracts_title_ngrams"
    ].apply(lambda toks: [replacements.get(t, t) for t in toks])

    # Final filtered tokens (remove rare/common/extra)
    df["filtered_3_token_processed_abstracts_title_ngrams"] = df[
        "filtered_2_token_processed_abstracts_title_ngrams"
    ].apply(lambda toks: filter_tokens(toks, tokens_to_remove))

    # Plot after removal
    # plot_token_freq(
    #     df["filtered_3_token_processed_abstracts_title_ngrams"],
    #     out_png=os.path.join(fig_dir, "token_freq_after_filter.png"),
    #     out_svg=os.path.join(fig_dir, "token_freq_after_filter.svg"),
    #     title_suffix="After Filtering",
    # )

    df.to_csv(output_csv, index=False)
    print(f"Saved processed topic-model data to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for topic modeling.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV.")
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to output processed CSV with filtered token column.",
    )
    parser.add_argument(
        "--fig_dir",
        default="figures/topic_model",
        help="Directory to save frequency plots.",
    )
    args = parser.parse_args()
    prepare_topic_data(args.input_csv, args.output_csv, args.fig_dir)


if __name__ == "__main__":
    main()
