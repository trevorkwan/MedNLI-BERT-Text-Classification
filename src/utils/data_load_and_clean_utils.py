import json
import random
import logging
import os
from datasets import load_dataset
from utils.args_utils import *
from config import RAW_TRAIN_FILE, RAW_VALIDATION_FILE, RAW_TEST_FILE, CLEAN_DATA_DIR, CLEAN_TRAIN_FILE, CLEAN_VALIDATION_FILE, CLEAN_TEST_FILE

# need to set logger in helper function scripts
logger = logging.getLogger(__name__)

# get args
data_args, model_args, training_args = parse_arguments()

def load_raw_data():
    """
    Load raw data from specified train, validation, and test files.

    Args:
        data_args: A DataLoadingArguments object containing file paths for train, validation, and test data.

    Returns:
        train_med, eval_med, test_med: Lists containing raw training, validation, and test data.
    """
    train_med = [json.loads(line) for line in open(RAW_TRAIN_FILE, 'r')]
    eval_med = [json.loads(line) for line in open(RAW_VALIDATION_FILE, 'r')]
    test_med = [json.loads(line) for line in open(RAW_TEST_FILE, 'r')]
    return train_med, eval_med, test_med

def get_clean_data(data, data_args, max_samples=None):
    """
    Clean and preprocess data by removing extra whitespaces and creating data_list with sentence pairs and labels.

    Args:
        data: A list of raw data dictionaries containing 'sentence1', 'sentence2', and 'gold_label' keys.
        max_samples (Optional[int]): The maximum number of samples to return from the data list. If not specified, all samples are returned.

    Returns:
        data_list: A list of dictionaries containing 'sentence1', 'sentence2', and 'label' keys.
    """

    s1 = [item[data_args.sentence1_key].strip() for item in data] # .strip() removes whitespace from a string (e.g, spaces, tabs)
    s2 = [item[data_args.sentence2_key].strip() for item in data]
    labels = [item[data_args.label_key] for item in data]

    data_list = [{"sentence1": s1[i], "sentence2": s2[i], "label": labels[i]} for i in range(len(s1))]

    if max_samples is not None:
        max_samples = min(len(data_list), max_samples)
        indices = list(range(len(data_list)))
        random.shuffle(indices)
        data_list = [data_list[i] for i in indices[:max_samples]]

    return data_list

def load_and_clean_data(logger, data_args):
    """
    Load and clean/preprocess data from specified train, validation, and test files.

    Args:
        data_args: A DataLoadingArguments object containing file paths for train, validation, and test data.

    Returns:
        train_list, eval_list, test_list: Lists of cleaned and preprocessed training, validation, and test data.
    """
    logger.info(
        f"Starting to load data from files: {data_args.train_file, data_args.validation_file, data_args.test_file}"
    )

    train_med, eval_med, test_med = load_raw_data()

    train_list = get_clean_data(train_med, data_args, data_args.max_train_samples)
    logger.info(f"Loading training data with sample size: {len(train_list)}.")

    eval_list = get_clean_data(eval_med, data_args, data_args.max_eval_samples)
    logger.info(f"Loading evaluation data with sample size: {len(eval_list)}.")

    test_list = get_clean_data(test_med, data_args, data_args.max_test_samples)
    logger.info(f"Loading testing data with sample size: {len(test_list)}.")

    return train_list, eval_list, test_list

def export_json(data, file_path):
    """
    Export data to a JSON file.

    Args:
        data: Data to be exported as JSON.
        file_path: The path to the file where the JSON data will be saved.
    """
    with open(file_path, "w") as outfile:
        json.dump(data, outfile)

def export_clean_data(logger, train_list, eval_list, test_list, os):
    """
    Export cleaned and preprocessed train, validation, and test data to JSON files.

    Args:
        train_list: A list of cleaned and preprocessed training data.
        eval_list: A list of cleaned and preprocessed evaluation data.
        test_list: A list of cleaned and preprocessed test data.
    """
    export_json(train_list, os.path.join(CLEAN_DATA_DIR, "clean_train_med.json"))
    export_json(eval_list, os.path.join(CLEAN_DATA_DIR, "clean_eval_med.json"))
    export_json(test_list, os.path.join(CLEAN_DATA_DIR, "clean_test_med.json"))

    logger.info(f"Finished exporting cleaned data as json files.")

def get_data_files(data_args):
    """
    Gets the data file paths for train, validation, and test sets.

    Args:
        data_args (DataTrainingArguments): An instance of DataTrainingArguments containing the data arguments.

    Returns:
        dict: A dictionary containing the file paths for train, validation, and test sets.
    """

    data_files = {
        "train": CLEAN_TRAIN_FILE,
        "validation": CLEAN_VALIDATION_FILE,
        "test": CLEAN_TEST_FILE,
    }
    return data_files

def load_clean_datasets(logger, data_files, model_args, data_args):
    """
    Loads clean datasets from the provided data files.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        data_files (dict): A dictionary containing the file paths for train, validation, and test sets.
        model_args (ModelArguments): An instance of ModelArguments containing the model arguments.
        data_args (DataTrainingArguments): An instance of DataTrainingArguments containing the data arguments.

    Returns:
        datasets.DatasetDict: A dictionary containing the clean datasets for train, validation, and test sets.
    """
    logger.info(f"Loading clean datasets...")

    if data_args.train_file.endswith(".csv"):
        clean_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        clean_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    return clean_datasets