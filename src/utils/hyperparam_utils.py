import logging
import json
import os
import optuna

from transformers import (
    Trainer,
    EarlyStoppingCallback
)

from utils.args_utils import *
from utils.data_preprocessing_utils import *
from utils.model_utils import *

from config import OPTIMIZED_HYPERPARAMS_DIR

# need to set logger in helper function scripts
logger = logging.getLogger(__name__)

def create_study(logger):
    """
    Creates an Optuna study for hyperparameter optimization.

    Args:
        logger (logging.Logger): A logger object for logging messages.

    Returns:
        optuna.study.Study: An Optuna study object for hyperparameter optimization.
    """
    logger.info(f"Creating study...")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), study_name="Optimize Hyperparameters")
    study.enqueue_trial({"learning_rate": 3e-5, "per_device_train_batch_size": 8, "num_train_epochs": 3, "max_seq_length": 128, "weight_decay": 1e-2})
    return study

def get_search_space(param_name):
    """
    Defines the search space for hyperparameter optimization.

    Args:
        param_name (str): The name of the hyperparameter for which to get the search space.

    Returns:
        tuple: A tuple containing the search space for the specified hyperparameter.
    """
    search_space = {
        "learning_rate": (2e-5, 5e-5),
        "per_device_train_batch_size": (8, 16),
        "num_train_epochs": (2, 4),
        "max_seq_length": (128, 384),
        "weight_decay": (1e-4, 1e-2)
    }
    return search_space[param_name]

def get_hyperparameters(trial, get_search_space, logger):
    """
    Gets the hyperparameters to try from the search space and logs the current trial number and hyperparameter values.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object for the current trial.
        get_search_space (function): A function to get the search space for a specific hyperparameter.
        logger (logging.Logger): A logger object for logging messages.

    Returns:
        tuple: A tuple containing the suggested hyperparameter values.
    """
    learning_rate = trial.suggest_loguniform("learning_rate", *get_search_space("learning_rate")) # e.g. trial.suggest_loguniform() accesses the trainer object and modifies its learning rate
    per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", *get_search_space("per_device_train_batch_size"))
    num_train_epochs = trial.suggest_int("num_train_epochs", *get_search_space("num_train_epochs"))
    max_seq_length = trial.suggest_int("max_seq_length", *get_search_space("max_seq_length"), step=64)
    weight_decay = trial.suggest_loguniform("weight_decay", *get_search_space("weight_decay"))
    
    logger.info(f"Starting trial {trial.number} with learning_rate = {learning_rate:.5f}, per_device_train_batch_size = {per_device_train_batch_size}, num_train_epochs = {num_train_epochs}, max_seq_length = {max_seq_length}, weight_decay = {weight_decay:.5f}.")
    
    return learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay

def update_training_args_with_hyperparams(logger, training_args, learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay):
    """
    Sets the training arguments to the suggested hyperparameter values and configures early stopping.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.
        learning_rate (float): The suggested learning rate value.
        per_device_train_batch_size (int): The suggested per-device train batch size value.
        num_train_epochs (int): The suggested number of training epochs.
        max_seq_length (int): The suggested maximum sequence length value.
        weight_decay (float): The suggested weight decay value.

    Returns:
        MyTrainingArguments: The updated instance of MyTrainingArguments with the suggested hyperparameter values.
    """
    logger.info(f"Setting training_args attributes to suggested values.")

    # feed the suggested values into training_args
    setattr(training_args, "learning_rate", learning_rate)
    setattr(training_args, "per_device_train_batch_size", per_device_train_batch_size)
    setattr(training_args, "num_train_epochs", num_train_epochs)
    setattr(training_args, "max_seq_length", max_seq_length)
    setattr(training_args, "weight_decay", weight_decay)
    return training_args

def update_training_args_with_eval_steps(logger, training_args, clean_datasets):
    """
    Sets the training arguments to the suggested hyperparameter values and configures early stopping.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.
    Returns:
        MyTrainingArguments: The updated instance of MyTrainingArguments with the suggested hyperparameter values.
    """
    logger.info(f"Setting training_args attributes to suggested values.")
    
    train_length = len(clean_datasets["train"])
    setattr(training_args, "eval_steps", int(train_length/training_args.per_device_train_batch_size*(training_args.num_times_eval_per_epoch**-1))) # if num_times_eval_per_epoch = 2, then 2^-1 = 0.5
    # train_length / training_args.per_device_train_batch_size = number of training steps per epoch
    # number of training steps per epoch * (training_args.num_times_eval_per_epoch**-1) = number of training steps between evaluations
    return training_args

def train_and_evaluate(trial, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets):
    """
    Trains and evaluates the model with the given hyperparameters and datasets.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object that contains trial information and hyperparameters.
        data_args (DataArguments): The data arguments containing dataset-related configurations.
        model_args (ModelArguments): The model arguments containing model-related configurations.
        training_args (TrainingArguments): The training arguments containing training-related configurations.
        logger (logging.Logger): The logger to display information during the process.
        last_checkpoint (str): The path to the last checkpoint from which to resume training.
        clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.

    Returns:
        float: The accuracy of the model on the evaluation dataset.
    """
    
    # get hyperparameters for the current trial
    learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay = get_hyperparameters(trial, get_search_space, logger)
    
    # update training arguments with the trial hyperparameters
    training_args = update_training_args_with_hyperparams(logger, training_args, learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay)

    # get label information from the dataset
    label_list, num_labels = get_label_information(logger, clean_datasets)

    # load pre-trained config, tokenizer, and model
    config, tokenizer, model = load_config_tokenizer_model(logger, model_args, num_labels)
    
    # update max_seq_length if it is too long for the tokenizer
    max_seq_length = get_max_seq_length(max_seq_length, tokenizer, logger)

    # preprocess datasets to get sentence1_key, sentence2_key, and label_to_id
    sentence1_key, sentence2_key, label_to_id = preprocess_clean_datasets(logger, config, num_labels, label_list, data_args)
    
    # preprocess datasets by tokenizing and truncating
    clean_datasets = preprocess_datasets(logger, clean_datasets, training_args, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer)

    # update training arguments with early stopping
    training_args = update_training_args_with_eval_steps(logger, training_args, clean_datasets)
    
    # get train and eval subsets
    short_train, short_eval = get_subsets(logger, clean_datasets, data_args, training_args)

    # create data collator for padding and batching
    data_collator = create_data_collator(logger, training_args, tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=short_train,
        eval_dataset=short_eval,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    logger.info(f"*** Training Trial {trial.number} ***")
    
    # determine the checkpoint to resume training from
    checkpoint = last_checkpoint if last_checkpoint is not None else training_args.resume_from_checkpoint

    # train the model (fine-tune)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # get training metrics
    metrics = train_result.metrics
    # store the number of training samples
    metrics["train_samples"] = len(short_train)

    logger.info(f"*** Evaluating Trial {trial.number} ***")

    # evaluate the model
    metrics = trainer.evaluate(eval_dataset=short_eval)
    # store the number of evaluation samples
    metrics["eval_samples"] = len(short_eval)

    logger.info(f"Trial {trial.number} finished with accuracy: {metrics['eval_accuracy']:.4f}.")

    return metrics["eval_accuracy"]

# define the hyperparameter optimization function
def objective(trial, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets):
    """
    The objective function for the hyperparameter optimization process.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object that contains trial information and hyperparameters.
        data_args (DataArguments): The data arguments containing dataset-related configurations.
        model_args (ModelArguments): The model arguments containing model-related configurations.
        training_args (TrainingArguments): The training arguments containing training-related configurations.
        logger (logging.Logger): The logger to display information during the process.
        last_checkpoint (str): The path to the last checkpoint from which to resume training.
        clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.

    Returns:
        float: The accuracy of the model on the evaluation dataset.
    """
    return train_and_evaluate(
            trial=trial,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            logger=logger,
            last_checkpoint=last_checkpoint,
            clean_datasets=clean_datasets
        )

def optimize_hyperparameters(study, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets):
    """
    Optimizes the hyperparameters using the objective function and Optuna.

    Args:
        study (optuna.study.Study): An Optuna study object to manage the optimization process.
        data_args (DataArguments): The data arguments containing dataset-related configurations.
        model_args (ModelArguments): The model arguments containing model-related configurations.
        training_args (TrainingArguments): The training arguments containing training-related configurations.
        logger (logging.Logger): The logger to display information during the process.
        last_checkpoint (str): The path to the last checkpoint from which to resume training.
        clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
    """
    study.optimize(lambda trial: objective(trial, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets),
                   n_trials=training_args.n_trials)
    return study

def save_optimized_hyperparams(logger, study):
    """
    Saves the optimized hyperparameters as a JSON file.

    Args:
        logger (logging.Logger): The logger to display information during the process.
        study (optuna.study.Study): An Optuna study object to manage the optimization process.
        output_path (str): The path to the output file where the optimized hyperparameters will be saved.
    """
    logger.info(f"Saving optimized hyperparameters...")

    optimized_hyperparams = {
        'best_hyperparameters': study.best_params,
        'best_accuracy': study.best_value
    }
    output_path = os.path.join(OPTIMIZED_HYPERPARAMS_DIR, 'optimized_hyperparameters.json')

    with open(output_path, 'w') as f:
        json.dump(optimized_hyperparams, f)

def load_optimized_hyperparameters(logger):
    """
    Load optimized hyperparameters from a JSON file if the file exists in the given folder.
    If the file or folder is not found, a warning is logged, and the function returns None.

    Args:
        logger: Logger instance for logging information and warnings.

    Returns:
        dict: A dictionary containing the optimized hyperparameters or None if the file or folder is not found.
    """
    filepath = OPTIMIZED_HYPERPARAMS_DIR + "optimized_hyperparameters.json"

    if not os.path.exists(filepath):
        logger.warning(f"No optimized hyperparameters file found at {filepath}. File path does not exist.")
        logger.info(f"Using specified and default hyperparameters instead.")
        return None

    folder_path, _ = os.path.split(filepath)
    if not os.listdir(folder_path):
        logger.warning(f"No files found in the folder: {folder_path}. Files do not exist.")
        logger.info(f"Using specified and default hyperparameters instead.")
        return None
    
    else:
        logger.info(f"Loading optimized hyperparameters from {filepath}...")

        with open(filepath, "r") as f:
            optimized_hyperparameters = json.load(f)
    return optimized_hyperparameters

def set_optimized_hyperparameters(logger, training_args, optimized_hyperparameters):
    """
    Update the training arguments with optimized hyperparameters or user-specified/default values.
    If optimized_hyperparameters is None or training_args.use_optimized_hyperparams is set to False,
    the function will use specified and default hyperparameters.

    Args:
        logger: Logger instance for logging information and warnings.
        training_args (TrainingArguments): The training arguments containing training-related configurations.
        optimized_hyperparameters (dict): A dictionary containing the optimized hyperparameters or None.

    Returns:
        TrainingArguments: The updated training arguments with optimized hyperparameters or user-specified/default values.
    """

    if optimized_hyperparameters is None:
        logger.info(f"Using specified and default hyperparameters instead.")
        return training_args

    else:
        logger.info(f"Setting training_args attributes to optimized hyperparameter values...")

        for k, v in optimized_hyperparameters.items():
            if k == "best_hyperparameters":
                for k2, v2 in v.items():
                    if training_args.use_optimized_hyperparams: # use_optimized_hyperparams is default True, if False, ignores all optimized hyperparameters and uses default/specified values
                        # if the user did not change the hyperparam, then set it to the optimized value
                        setattr(training_args, k2, v2)
                    else:
                        # use the specified value
                        logger.info(f"Using specified value for {k2}: {getattr(training_args, k2)}")
        return training_args
