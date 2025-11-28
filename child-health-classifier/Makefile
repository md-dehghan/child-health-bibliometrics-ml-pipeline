# Paths (adjust filenames as needed)
ABBREV_INPUT = data/raw/Pub_df_make_abbreviation.csv
ABBREV_ORIG  = data/interim/abbreviation_dicts_abs.json
ABBREV_UPD   = data/interim/updated_abbreviation_dicts_abs.json

PROCESSED_TOKENS = data/processed/Pubs_processed_tokens.csv

MODEL_FILE = models/child_health_xgb_pipeline.pkl
METRICS_DIR = models/model_output

PREDICTIONS = data/processed/Pubs_with_predictions.csv

.PHONY: all abbrev process train train_hyper predict clean

# Run the full pipeline with default model
all: abbrev process train predict

# 1) Build abbreviation dictionaries
abbrev: $(ABBREV_ORIG) $(ABBREV_UPD)

$(ABBREV_ORIG) $(ABBREV_UPD): $(ABBREV_INPUT)
	python scripts/make_abbreviation_dicts.py \
	    --input_csv $(ABBREV_INPUT) \
	    --out_original_json $(ABBREV_ORIG) \
	    --out_updated_json $(ABBREV_UPD)

# 2) Process text
process: $(PROCESSED_TOKENS)

$(PROCESSED_TOKENS): $(ABBREV_UPD)
	python scripts/make_processed_text.py \
	    --input_csv $(ABBREV_INPUT) \
	    --abbrev_json $(ABBREV_UPD) \
	    --output_csv $(PROCESSED_TOKENS)

# 3) Train model with default settings
train: $(MODEL_FILE)

$(MODEL_FILE): $(PROCESSED_TOKENS)
	python scripts/train_model.py \
	    --labels_csv data/raw/Pubs_labeled.csv \
	    --features_csv $(PROCESSED_TOKENS) \
	    --output_model $(MODEL_FILE) \
	    --test_size 0.2 \
	    --random_state 42 \
	    --metrics_dir $(METRICS_DIR)

# 3b) Alternative: train with hyperparameter search
train_hyper:
	python scripts/train_model_hyperparam.py \
	    --labels_csv data/raw/Pubs_labeled.csv \
	    --features_csv $(PROCESSED_TOKENS) \
	    --output_model $(MODEL_FILE) \
	    --test_size 0.2 \
	    --random_state 42 \
	    --metrics_dir $(METRICS_DIR)

# 4) Predict labels on the full dataset
predict: $(PREDICTIONS)

$(PREDICTIONS): $(MODEL_FILE) $(PROCESSED_TOKENS)
	python scripts/predict_labels.py \
	    --features_csv $(PROCESSED_TOKENS) \
	    --model_path $(MODEL_FILE) \
	    --output_csv $(PREDICTIONS)

clean:
	rm -f $(ABBREV_ORIG) $(ABBREV_UPD) $(PROCESSED_TOKENS) $(MODEL_FILE)
	rm -rf $(METRICS_DIR)
