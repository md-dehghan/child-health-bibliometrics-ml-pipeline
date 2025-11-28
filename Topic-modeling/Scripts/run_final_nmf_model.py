#!/usr/bin/env python
"""
Fit final NMF topic model with chosen n_topics (e.g. 80),
compute coherence, export topics & document-topic distributions,
merge selected topics, and generate word clouds.

Usage:
    python scripts/run_final_nmf_model.py \
        --input_csv data/processed/CHW_pubs_topic_tokens.csv \
        --token_col filtered_3_token_processed_abstracts_title_ngrams \
        --n_topics 80 \
        --output_dir data/topic_model \
        --fig_dir figures/topic_model
"""

import argparse
import ast
import os
from typing import Any, List, Dict

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


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
    processed_docs = []
    texts = []
    for tokens in tokens_series:
        toks = safe_eval_list(tokens)
        joined = ["_".join(str(t).split()) for t in toks]
        processed_docs.append(" ".join(joined))
        texts.append(joined)
    return processed_docs, texts


def get_topics(model: NMF, vectorizer: TfidfVectorizer, top_n: int = 10):
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic in model.components_:
        top_idx = topic.argsort()[:-top_n - 1:-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def get_topics_with_weights(model: NMF, vectorizer: TfidfVectorizer, top_n: int = 10):
    feature_names = vectorizer.get_feature_names_out()
    topics_with_weights = []
    for topic in model.components_:
        top_idx = topic.argsort()[:-top_n - 1:-1]
        topics_with_weights.append([(feature_names[i], topic[i]) for i in top_idx])
    return topics_with_weights


def merge_topics(W: np.ndarray, H: np.ndarray, merge_pairs_1based: List[List[int]]):
    """
    Merge topics in W/H according to lists of 1-based indices, e.g. [[12, 35], [11, 70], ...].

    Returns:
        W_merged, H_merged
    """
    merge_pairs = [[i - 1 for i in group] for group in merge_pairs_1based]
    index_map = {i: i for i in range(H.shape[0])}
    H_new = H.copy()
    W_new = W.copy()

    for group in merge_pairs:
        current_indices = [index_map[i] for i in group]

        # Merge in H (topics vs terms)
        merged_topic = np.sum(H_new[current_indices, :], axis=0)
        H_new[current_indices[0], :] = merged_topic
        H_new = np.delete(H_new, current_indices[1:], axis=0)

        # Merge in W (docs vs topics)
        merged_doc_topic = np.sum(W_new[:, current_indices], axis=1)
        W_new[:, current_indices[0]] = merged_doc_topic
        W_new = np.delete(W_new, current_indices[1:], axis=1)

        removed_indices = current_indices[1:]
        index_map = {
            orig: new_idx - sum(1 for r in removed_indices if r < new_idx)
            for orig, new_idx in index_map.items()
            if new_idx not in removed_indices
        }

    return W_new, H_new


def color_func(word, font_size, position, orientation, font_weight, random_state=None, **kwargs):
    """Color words along a cold-to-warm spectrum based on weight."""
    red = int(255 * font_weight)
    green = int(128 * (1 - font_weight))
    blue = int(255 * (1 - font_weight))
    return f"rgb({red}, {green}, {blue})"


def plot_word_clouds_for_updated_topics(
    H: np.ndarray,
    vectorizer: TfidfVectorizer,
    title_topics: Dict[str, str],
    fig_path: str,
    num_words: int = 10,
    scale_factor: float = 5.0,
    min_weight: float = 0.1,
    use_log: bool = True,
):
    """
    Plot word clouds for updated topics (rows of H).

    - Word sizes are uniform; color encodes relative weight.
    """
    num_topics = H.shape[0]
    feature_names = vectorizer.get_feature_names_out()

    n_cols = 5
    n_rows = (num_topics // n_cols) + (num_topics % n_cols > 0)

    matplotlib.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * 4, n_rows * 3),
    )
    axes = axes.flatten()

    for topic_idx, ax in enumerate(axes):
        if topic_idx < num_topics:
            top_idx = H[topic_idx].argsort()[:-num_words - 1:-1]
            top_words = [feature_names[i] for i in top_idx]
            top_weights = H[topic_idx][top_idx]

            max_weight = max(top_weights) if max(top_weights) > 0 else 1.0
            normalized = top_weights / max_weight

            if use_log:
                normalized = np.log1p(normalized)

            adjusted = np.clip(normalized * scale_factor, min_weight, 1.0)

            uniform_freq = {w: 1 for w in top_words}
            word_colors = {w: float(wt) for w, wt in zip(top_words, adjusted)}

            wc = WordCloud(
                width=1300,
                height=400,
                background_color="white",
                max_font_size=100,
                relative_scaling=0,
                color_func=lambda word, *args, **kwargs: color_func(
                    word, font_weight=word_colors[word], *args, **kwargs
                ),
            ).generate_from_frequencies(uniform_freq)

            topic_title = title_topics.get(f"Topic {topic_idx + 1}", f"Topic {topic_idx + 1}")
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(topic_title, fontsize=10, fontweight="bold", pad=8)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(fig_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved word clouds to: {fig_path}")


# ---------------------------------------------------------------------
# Topic titles mapping (as in your notebook)
# ---------------------------------------------------------------------

TITLE_TOPICS = {
    "Topic 1": "Environment in Growth and Development",
    "Topic 2": "Genetic and Environment interaction",
    "Topic 3": "Pregnancy, Gestational Health, and Postpartum",
    "Topic 4": "Autism Spectrum Disorders",
    "Topic 5": "COVID-19 Pandemic",
    "Topic 6": "Pain Management and Analgesic Interventions",
    "Topic 7": "Parental Perception and Communication in the NICU",
    "Topic 8": "Stem Cell Differentiation and Proliferation",
    "Topic 9": "Stress, Trauma, and Resilience",
    "Topic 10": "Depression in Postpartum",
    "Topic 11": "Infant Feeding and NICU care",
    "Topic 12": "Physical Activity and Sedentary Behavior",
    "Topic 13": "Vaccine and Immunization",
    "Topic 14": "Cancers",
    "Topic 15": "Sleep Disturbances and Disorders",
    "Topic 16": "Brain and Cognitive Connectivity",
    "Topic 17": "School Environment and Bullying",
    "Topic 18": "Sex, Abuse, and Violence in Relationships and Minority",
    "Topic 19": "Attention Deficit Hyperactivity Disorder",
    "Topic 20": "Epilepsy and Seizures Disorder",
    "Topic 21": "Mental Health, Substance Use, Well-being in Vulnerable Populations",
    "Topic 22": "Asthma and Allergies",
    "Topic 23": "Social Communication, Resilience, and Relationships",
    "Topic 24": "Maternal Health, Childbirth, and Outcomes",
    "Topic 25": "Intervention, Nutrition, and Exercise",
    "Topic 26": "Food Allergies, Insecurity, Nutrition and Environment",
    "Topic 27": "Obesity and Gestational Weight Gain",
    "Topic 28": "Heart Disease",
    "Topic 29": "Syndromic Disorders and Metabolism",
    "Topic 30": "Suicide and Bullying",
    "Topic 31": "Cannabis, Tobacco, and Alcohol",
    "Topic 32": "Diet, Nutrition, and Metabolism",
    "Topic 33": "Gut Microbiota, Immune System, and Inflammation",
    "Topic 34": "Family Dynamic and Socioeconomic Resilience in Immigrant & Refugee",
    "Topic 35": "Surgery, Postoperative Care, and Preoperative Interventions",
    "Topic 36": "Infection, Antibiotic, and Immune Responses",
    "Topic 37": "Genetic Mutations and Disorders",
    "Topic 38": "Athlete Training and Sport Psychology",
    "Topic 39": "Pediatric Guidelines and Consensus in Medicine",
    "Topic 40": "Genetic Variants and Neurodevelopment",
    "Topic 41": "Community Engagement and Healthcare Policy",
    "Topic 42": "Concussion and Recovery in Athletes",
    "Topic 43": "Student Learning and Classroom Engagement",
    "Topic 44": "Tumors and Treatment",
    "Topic 45": "Behavioral Patterns and Social Interaction",
    "Topic 46": "Infant Feeding and Postpartum",
    "Topic 47": "Bone Health and Disorders",
    "Topic 48": "Anxiety and Distress in Postpartum and Pandemic",
    "Topic 49": "Inflammatory Bowel Disease",
    "Topic 50": "Language Development",
    "Topic 51": "Cystic Fibrosis and Respiratory Complications",
    "Topic 52": "Caregiving, Distress, and Dementia",
    "Topic 53": "Muscle Health and Disorders",
    "Topic 54": "Fetal Alcohol Spectrum Disorder",
    "Topic 55": "Reading, Literacy, and Comprehension",
    "Topic 56": "Indigenous Community, Identity, and Culture",
    "Topic 57": "Postpartum Parenting",
    "Topic 58": "Mitochondrial Function and Metabolism",
    "Topic 59": "Economic Costs, Healthcare, Policy, and Hospitalization",
    "Topic 60": "Vitamins, Supplements, and Deficiencies",
    "Topic 61": "Bipolar Disorder, Schizophrenia, and Mood Disorders",
    "Topic 62": "Multiple Sclerosis and Immune Function",
    "Topic 63": "Training and Learning in Healthcare",
    "Topic 64": "Kidney Injury, Disease, and Transplant",
    "Topic 65": "Cerebral Palsy, Motor Function, and Rehabilitation",
    "Topic 66": "RNA and Genome Sequencing",
    "Topic 67": "Education, Income, and Socioeconomic Inequality Among Immigrants",
    "Topic 68": "Prenatal, Postnatal, and Stroke",
    "Topic 69": "Healthcare and Socioeconomic Factors in Mortality and Morbidity",
    "Topic 70": "Disability, Intellectual Development, and Rehabilitation",
    "Topic 71": "Gestational Diabetes",
    "Topic 72": "TBI: Injury and Cognitive Function",
    "Topic 73": "Emergency Department and Hospitalization",
    "Topic 74": "Placental Function and Fetal Growth",
}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def run_final_nmf_model(
    input_csv: str,
    token_col: str,
    n_topics: int,
    output_dir: str,
    fig_dir: str,
    random_state: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    if token_col not in df.columns:
        raise ValueError(f"Column '{token_col}' not found in input CSV.")

    corpus, texts = tokens_to_corpus(df[token_col])
    dictionary = Dictionary(texts)
    gensim_corpus = [dictionary.doc2bow(t) for t in texts]

    # Vectorize for NMF
    tfidf_vectorizer = TfidfVectorizer()
    dtm_nmf = tfidf_vectorizer.fit_transform(corpus)

    nmf_model = NMF(n_components=n_topics, max_iter=1000, random_state=random_state)
    W = nmf_model.fit_transform(dtm_nmf)
    H = nmf_model.components_
    reconstruction_error = nmf_model.reconstruction_err_

    # Topics & coherence
    nmf_topics = get_topics(nmf_model, tfidf_vectorizer, top_n=10)
    coherence_model = CoherenceModel(
        topics=nmf_topics, texts=texts, dictionary=dictionary, coherence="c_v"
    )
    nmf_coherence_score = coherence_model.get_coherence()
    nmf_topic_coherence = coherence_model.get_coherence_per_topic()

    print(f"NMF overall coherence (c_v): {nmf_coherence_score:.4f}")
    print(f"Reconstruction error: {reconstruction_error:.4f}")

    nmf_topics_with_weights = get_topics_with_weights(nmf_model, tfidf_vectorizer, top_n=10)

    topics_data = {
        "Topic Number": [f"Topic {i + 1}" for i in range(n_topics)],
        "Top 10 Keywords with Weights": [
            ", ".join(f"{w} ({wt:.4f})" for w, wt in topic)
            for topic in nmf_topics_with_weights
        ],
        "Coherence Score": nmf_topic_coherence,
    }
    topics_weight_df = pd.DataFrame(topics_data)

    topics_weight_path = os.path.join(
        output_dir, f"nmf_topics_with_coherence_topicn_{n_topics}.csv"
    )
    topics_weight_df.to_csv(topics_weight_path, index=False)
    print(f"Saved topics + coherence to: {topics_weight_path}")

    # Simple topic summary (without weights)
    nmf_topic_df = pd.DataFrame(
        {
            "Topic Number": [f"Topic {i + 1}" for i in range(n_topics)],
            "Coherence Score": nmf_topic_coherence,
            "Top Words": [", ".join(t) for t in nmf_topics],
        }
    )
    topic_summary_path = os.path.join(
        output_dir, f"nmf_topic_summary_topicn_{n_topics}.csv"
    )
    nmf_topic_df.to_csv(topic_summary_path, index=False)
    print(f"Saved topic summary to: {topic_summary_path}")

    # Document-topic distributions
    topic_distribution_df = pd.DataFrame(
        W, columns=[f"Topic_{i + 1}" for i in range(W.shape[1])]
    )
    if "EID" in df.columns:
        topic_distribution_df["EID"] = df["EID"].values
    if "title" in df.columns:
        topic_distribution_df["title"] = df["title"].values

    topic_cols = [c for c in topic_distribution_df.columns if c.startswith("Topic_")]
    topic_distribution_df["Total_Topic_Weight"] = topic_distribution_df[topic_cols].sum(
        axis=1
    )

    doc_topic_path = os.path.join(
        output_dir, f"nmf_topic_distributions_topicn_{n_topics}.csv"
    )
    topic_distribution_df.to_csv(doc_topic_path, index=False)
    print(f"Saved document-topic distributions to: {doc_topic_path}")

    # Merge topics as in your notebook
    merge_pairs = [
        [12, 35],
        [11, 70],
        [19, 79],
        [76, 21],
        [14, 16],
        [77, 22],
    ]
    W_merged, H_merged = merge_topics(W, H, merge_pairs)
    print("Original H shape:", H.shape)
    print("Merged   H shape:", H_merged.shape)
    print("Original W shape:", W.shape)
    print("Merged   W shape:", W_merged.shape)

    # Save merged matrices
    np.save(os.path.join(output_dir, f"W_merged_topicn_{n_topics}.npy"), W_merged)
    np.save(os.path.join(output_dir, f"H_merged_topicn_{n_topics}.npy"), H_merged)

    # Save model and vectorizer
    joblib.dump(
        nmf_model, os.path.join(output_dir, f"nmf_model_topicn_{n_topics}.pkl")
    )
    joblib.dump(
        tfidf_vectorizer,
        os.path.join(output_dir, f"tfidf_vectorizer_topicn_{n_topics}.pkl"),
    )

    # Word clouds for merged topics
    wordcloud_path = os.path.join(
        fig_dir, f"wordcloud_topics_merged_topicn_{n_topics}.svg"
    )
    plot_word_clouds_for_updated_topics(
        H_merged,
        tfidf_vectorizer,
        TITLE_TOPICS,
        fig_path=wordcloud_path,
        num_words=10,
        scale_factor=5.0,
        min_weight=0.1,
        use_log=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run final NMF model with chosen number of topics."
    )
    parser.add_argument("--input_csv", required=True, help="Processed input CSV.")
    parser.add_argument(
        "--token_col",
        default="filtered_3_token_processed_abstracts_title_ngrams",
        help="Column containing final token lists.",
    )
    parser.add_argument("--n_topics", type=int, default=80, help="Number of topics.")
    parser.add_argument(
        "--output_dir",
        default="data/topic_model",
        help="Directory to save model outputs.",
    )
    parser.add_argument(
        "--fig_dir",
        default="figures/topic_model",
        help="Directory to save plots (word clouds, etc.).",
    )
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    run_final_nmf_model(
        input_csv=args.input_csv,
        token_col=args.token_col,
        n_topics=args.n_topics,
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
