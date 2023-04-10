import logging # logs msgs/errors 

from transformers import (
    set_seed # setting a random seed
)

from transformers.utils import check_min_version # checks if transformers package meets minimum version requirements

from utils.args_utils import *
from utils.logging_utils import *
from utils.checkpoint_utils import *
from utils.data_load_and_clean_utils import *
from utils.data_preprocessing_utils import *
from utils.hyperparam_utils import *
from utils.model_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

def main():

    # Parse arguments
    data_args, model_args, training_args = parse_arguments()

    # Setup logging
    logger = logging.getLogger(__name__) # creates a logger object with the "name" get_best_model, to identify where these logger messages are coming from
    logger = setup_logging(logger, training_args)

    # Detect last checkpoint
    last_checkpoint = detect_last_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get data files
    data_files = get_data_files(data_args)

    # Log data files
    log_data_files(logger, data_files)

    # Load clean datasets
    clean_datasets = load_clean_datasets(logger, data_files, model_args, data_args)

    # Load optimized hyperparameters
    optimized_hyperparameters = load_optimized_hyperparameters(logger)

    # Set the optimized hyperparameters to the training arguments
    training_args = set_optimized_hyperparameters(logger, training_args, optimized_hyperparameters)

    # Update training arguments with early stopping
    training_args = update_training_args_with_eval_steps(logger, training_args, clean_datasets)
    
    # Get label information
    label_list, num_labels = get_label_information(logger, clean_datasets)
    
    # Load pre-trained config, tokenizer, and model
    config, tokenizer, model = load_config_tokenizer_model(logger, model_args, num_labels)

    # Get max_seq_length
    max_seq_length = get_max_seq_length(training_args.max_seq_length, tokenizer, logger)

    # Get non_label_column_names
    non_label_column_names = get_non_label_column_names(logger, clean_datasets)

    # Get sentence1_key, sentence2_key
    sentence1_key, sentence2_key = get_sentence_keys(logger, non_label_column_names)

    # Get label_to_id
    label_to_id = get_label_to_id(logger, config, num_labels, label_list)

    # Preprocess clean datasets
    sentence1_key, sentence2_key, label_to_id = preprocess_clean_datasets(logger, clean_datasets, config, num_labels, label_list)
    
    # Preprocess datasets by tokenizing and truncating
    clean_datasets = preprocess_datasets(logger, clean_datasets, training_args, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer)
    
    # Get train dataset
    train_dataset = clean_datasets["train"]

    # Get the eval dataset
    eval_dataset = clean_datasets["validation"]

    # Create data collator
    data_collator = create_data_collator(logger, training_args, tokenizer)

    # Log a few random samples from the training set
    log_random_samples(logger, train_dataset)

    # Create the best trainer with early stopping callback
    best_trainer = create_best_trainer(logger,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model using the best trainer
    metrics = train_model(logger, best_trainer, train_dataset, last_checkpoint)

    # Save the trainer's state, metrics, and model to disk
    save_trainer_state(logger, best_trainer, metrics)
    
if __name__ == "__main__": # if the script is run from the command line, sets __name__ to "__main__" and main() will be executed. if the script is imported into another script, `__name__` will be set to `get_best_model` and main() will not be executed
    main()