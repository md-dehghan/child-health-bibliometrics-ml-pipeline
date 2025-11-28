# Topic Modeling Pipeline for Child-Health Research

This folder contains the full topic-modeling pipeline applied to the **child-health–relevant** subset of publications identified by the classifier.

The goal of this pipeline is to:

- Clean and normalize n-gram tokens for titles and abstracts  
- Prepare an input dataset suitable for topic modeling  
- Explore a range of topic numbers using NMF (Non-negative Matrix Factorization)  
- Select a final number of topics based on multiple metrics  
- Fit the final NMF model, export topic-word lists, document–topic weights, and merged topics  
- Generate publication-ready plots and word clouds for interpretation

---

## Folder Structure

Assuming you are inside the `topic-modeling/` folder:

```text
topic-modeling/
├── README.md
├── Makefile
├── requirements.txt        # optional (or use repo-level requirements)
├── scripts/
│   ├── prepare_topic_model_dat.py
│   ├── find_optimal_nmf_topics.py
│   └── run_final_nmf_model.py
├── data/
│   ├── raw/
│   │   └── CHW_pubs.csv                        # input child-health–relevant data (not in repo)
│   ├── processed/
│   │   ├── CHW_pubs_topic_tokens.csv           # prepared tokens for topic modeling
│   │   └── nmf_topic_search_results.csv        # metrics over different topic numbers
│   └── topic_model/
│       ├── nmf_topics_with_coherence_topicn_XX.csv
│       ├── nmf_topic_summary_topicn_XX.csv
│       ├── nmf_topic_distributions_topicn_XX.csv
│       ├── W_merged_topicn_XX.npy
│       └── H_merged_topicn_XX.npy
└── figures/
    └── topic_model/
        ├── token_freq_before_filter.png / .svg
        ├── token_freq_after_filter.png / .svg
        ├── nmf_reconstruction_error_vs_topics.png / .svg
        ├── nmf_coherence_vs_topics.png / .svg
        └── wordcloud_topics_merged_topicn_XX.svg