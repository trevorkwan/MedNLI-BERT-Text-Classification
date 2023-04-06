import logging
import sys
import random
import transformers.utils.logging
import datasets.utils.logging

def set_logger_and_verbosity(logger=None):
    """
    Set up the logging configuration and verbosity levels for the main libraries used in the script.

    Args:
        training_args: A MyTrainingArguments object containing the training arguments for the script.
    """
    if logger is None:
        logger = logging.getLogger(__name__) # creates a logger object with the "name" get_best_model, to identify where these logger messages are coming from
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # log format
        datefmt="%m/%d/%Y %H:%M:%S", # datetime format
        handlers=[logging.StreamHandler(sys.stdout)], # where to send the log messages e.g. a file, the console
    )
    
    logger.setLevel(logging.INFO) # set the logging level for the logger object for the entire script

def setup_logging(training_args):
    """
    Configures logging settings for the script.

    Args:
        training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.

    Returns:
        logger (logging.Logger): A logger object for logging messages.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    return logger

def log_data_files(logger, data_files):
    """
    Logs the data file paths using the logger.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        data_files (dict): A dictionary containing the file paths for train, validation, and test sets.
    """
    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

def log_random_samples(logger, train_dataset, num_samples=3):
    """
    Logs a few random samples from the training set.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        train_dataset (datasets.Dataset): The pre-processed training dataset.
        num_samples (int, optional): The number of random samples to log. Defaults to 3.
    """
    for index in random.sample(range(len(train_dataset)), num_samples):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

def log_best_hyperparameters(study, logger):
    """
    Logs the best hyperparameters found during the optimization process.

    Args:
        study (optuna.study.Study): An Optuna study object to manage the optimization process.
        logger (logging.Logger): The logger to display information during the process.
    """
    best_param_str = ", ".join(
        f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in study.best_params.items()
    )
    logger.info(f"Hyperparameter optimization completed with best hyperparameters: {best_param_str} with an accuracy of {study.best_value:.4f} during the trial evaluation.")
