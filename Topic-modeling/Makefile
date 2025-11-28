.PHONY: all prepare search final clean

# Paths (adjust filenames as needed)
RAW_TOKENS_CSV      = data/raw/CHW_pubs.csv
TOPIC_TOKENS_CSV    = data/processed/CHW_pubs_topic_tokens.csv
TOPIC_SEARCH_CSV    = data/processed/nmf_topic_search_results.csv
TOPIC_OUTPUT_DIR    = data/topic_model
FIG_DIR             = figures/topic_model

# Chosen number of topics for final model (update based on your search results)
N_TOPICS            = 80

# -------------------------------------------------------------------
# Full pipeline: prepare data -> search N topics -> fit final model
# -------------------------------------------------------------------
all: prepare search final

# 1) Prepare data for topic modeling
prepare:
	python scripts/prepare_topic_model_dat.py \
		--input_csv $(RAW_TOKENS_CSV) \
		--output_csv $(TOPIC_TOKENS_CSV) \
		--fig_dir $(FIG_DIR)

# 2) Grid search over NMF topic numbers
search:
	python scripts/find_optimal_nmf_topics.py \
		--input_csv $(TOPIC_TOKENS_CSV) \
		--token_col filtered_3_token_processed_abstracts_title_ngrams \
		--metrics_csv $(TOPIC_SEARCH_CSV) \
		--fig_dir $(FIG_DIR) \
		--min_topics 5 \
		--max_topics 200 \
		--step 5

# 3) Run final NMF model with chosen N_TOPICS
final:
	python scripts/run_final_nmf_model.py \
		--input_csv $(TOPIC_TOKENS_CSV) \
		--token_col filtered_3_token_processed_abstracts_title_ngrams \
		--n_topics $(N_TOPICS) \
		--output_dir $(TOPIC_OUTPUT_DIR) \
		--fig_dir $(FIG_DIR)