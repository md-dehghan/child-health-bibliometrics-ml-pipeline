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

```text
topic-modeling/
├── README.md
├── Makefile
├── requirements.txt       
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
│   └── topic_model      # Topic model results

└── figures/
    └── topic_model   # Topic model results 
