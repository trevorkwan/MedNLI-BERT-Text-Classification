# Makefile

# Set the python interpreter to be used
PYTHON := python

# Define the python scripts to be executed
CLEAN_DATA_SCRIPT := src/clean_data.py
OPTIMIZE_SCRIPT := src/optimize.py
TRAIN_SCRIPT := src/train.py
EVAL_SCRIPT := src/eval.py
PREDICT_SCRIPT := src/predict.py

all: clean_data optimize train eval predict

clean_data:
	$(PYTHON) $(CLEAN_DATA_SCRIPT) --train_file mli_train_v1.jsonl \
                                --validation_file mli_dev_v1.jsonl \
                                --test_file mli_test_v1.jsonl \
				--sentence1_key sentence1 \
				--sentence2_key sentence2 \
				--label_key gold_label \
				--max_train_samples 100 --max_eval_samples 50 --max_test_samples 50

optimize: 
	$(PYTHON) $(OPTIMIZE_SCRIPT) --train_file clean_train_med.json \
                                --validation_file clean_eval_med.json \
                                --test_file clean_test_med.json \
				--sentence1_key sentence1 \
				--sentence2_key sentence2 \
				--label_key gold_label \
                                --model_name_or_path bert-base-cased \
                                --train_subset_perc 0.15 \
                                --eval_subset_perc 0.4 \
                                --n_trials 2 \
				--output_dir results/hyperparameter_optimization/ \
				--overwrite_output_dir True 

train: 
	$(PYTHON) $(TRAIN_SCRIPT) --train_file clean_train_med.json \
                            --validation_file clean_eval_med.json \
                            --test_file clean_test_med.json \
			    --sentence1_key sentence1 \
			    --sentence2_key sentence2 \
			    --label_key gold_label \
                            --model_name_or_path bert-base-cased \
			    --learning_rate 3e-5 \
			    --per_device_train_batch_size 8 \
			    --num_train_epochs 3 \
			    --max_seq_length 128 \
			    --weight_decay 1e-2 \
			    --use_optimized_hyperparams True \
                            --output_dir results/training/ \
                            --overwrite_output_dir True

eval: 
	$(PYTHON) $(EVAL_SCRIPT) --train_file clean_train_med.json \
                            --validation_file clean_eval_med.json \
                            --test_file clean_test_med.json \
			    --sentence1_key sentence1 \
			    --sentence2_key sentence2 \
			    --label_key gold_label \
                            --model_name_or_path ../results/training \
                            --output_dir results/evaluation/ \
                            --overwrite_output_dir True

predict: 
	$(PYTHON) $(PREDICT_SCRIPT) --train_file clean_train_med.json \
                                --validation_file clean_eval_med.json \
                                --test_file clean_test_med.json \
				--sentence1_key sentence1 \
				--sentence2_key sentence2 \
				--label_key gold_label \
                                --model_name_or_path ../results/training \
                                --output_dir results/inference/ \
                                --overwrite_output_dir True

.PHONY: all clean_data optimize train evaluate predict
