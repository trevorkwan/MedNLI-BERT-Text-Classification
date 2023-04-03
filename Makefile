# Makefile

# Set the python interpreter to be used
PYTHON := python

# Define the python scripts to be executed
CLEAN_DATA_SCRIPT := scripts/clean_data.py
OPTIMIZE_SCRIPT := scripts/optimize.py
TRAIN_SCRIPT := scripts/train.py
EVAL_SCRIPT := scripts/eval.py
PREDICT_SCRIPT := scripts/predict.py

all: clean_data optimize train eval predict

clean_data:
	$(PYTHON) $(CLEAN_DATA_SCRIPT) --train_file mli_train_v1.jsonl \
                                --validation_file mli_dev_v1.jsonl \
                                --test_file mli_test_v1.jsonl \
                                --output_dir ../data/clean/ 

optimize: 
	$(PYTHON) $(OPTIMIZE_SCRIPT) --train_file clean_train_med.json \
                                --validation_file clean_eval_med.json \
                                --test_file clean_test_med.json \
                                --model_name_or_path bert-base-cased \
                                --output_dir ../results/hyperparameter_optimization/ \
                                --use_mps_device True \
                                --train_subset_perc 0.15 \
                                --eval_subset_perc 0.4 \
                                --overwrite_output_dir True \
                                --n_trials 10

train: 
	$(PYTHON) $(TRAIN_SCRIPT) --train_file clean_train_med.json \
                            --validation_file clean_eval_med.json \
                            --test_file clean_test_med.json \
                            --model_name_or_path bert-base-cased \
                            --output_dir ../results/training/ \
                            --use_mps_device True \
                            --overwrite_output_dir True

eval: 
	$(PYTHON) $(EVAL_SCRIPT) --train_file clean_train_med.json \
                            --validation_file clean_eval_med.json \
                            --test_file clean_test_med.json \
                            --model_name_or_path ../results/training \
                            --output_dir ../results/evaluation/ \
                            --use_mps_device True \
                            --overwrite_output_dir True

predict: 
	$(PYTHON) $(PREDICT_SCRIPT) --train_file clean_train_med.json \
                                --validation_file clean_eval_med.json \
                                --predict_file clean_test_med.json \
                                --model_name_or_path ../results/training \
                                --output_dir ../results/inference/ \
                                --use_mps_device True \
                                --overwrite_output_dir True

.PHONY: all clean_data optimize train evaluate predict
