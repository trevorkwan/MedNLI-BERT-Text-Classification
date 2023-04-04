import logging # logs msgs/errors 
import sys # helps interact with operating system
import os
import random # helps generate random numbers
from dataclasses import dataclass, field # helps create functions that store data 
from typing import Optional # Optional helps to specify that the param can also be None

import datasets # can load datasets from hugging face

import transformers
from transformers import (
    HfArgumentParser, # a sub-class of `argparse.ArgumentParser`, helps configure arguments to be passed to command-line
    TrainingArguments, # pass this in to an instance of Trainer for argument configurations
    set_seed, # setting a random seed
)

from transformers.utils import check_min_version # checks if transformers package meets minimum version requirements
import json

@dataclass
class DataLoadingArguments:
    """
    Arguments pertaining to what data we are going to input our model with for training and evaluation.
    """

    train_file: Optional[str] = field( # specify path to training data
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field( # specify path to validation data
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field( # specify path to testing data
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    max_train_samples: Optional[int] = field( # used for quicker training
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncates the number of training examples to this"
            )
        },
    )
    max_eval_samples: Optional[int] = field( # for quicker validation
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this"
            )
        },
    )
    max_test_samples: Optional[int] = field( # for quicker testing
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of predictions to this."
            )
        },
    )

    def __post_init__(self): # does additional checks after defining params
        if self.train_file is None or self.validation_file is None or self.test_file is None:
            raise ValueError("Need a training/validation/test file.")
        else:
            train_extension = self.train_file.split(".")[-1] # gets the document type e.g. "jsonl", makes sure that it is
            assert train_extension in ["jsonl"], "`train_file` should be a jsonl file."
            validation_extension = self.validation_file.split(".")[-1] # e.g. "jsonl". makes sure that it is
            assert validation_extension == train_extension, "`validation_file` should have the same extension as `train_file`."

@dataclass
class MyTrainingArguments(TrainingArguments):
    """
    My training arguments
    """

    seed: int = field(
        default = 123,
        metadata = {"help": "the seed to set"}
    )

    output_dir: str = field(
        default=None,
        metadata={"help": "specify output directory for parser"}
    )
    
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

logger = logging.getLogger(__name__) # creates a logger object with the "name" get_best_model, to identify where these logger messages are coming from

# Get the current working directory
cwd = os.getcwd()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)

def set_logger_and_verbosity(training_args, logger=None):
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

def load_raw_data(data_args):
    """
    Load raw data from specified train, validation, and test files.

    Args:
        data_args: A DataLoadingArguments object containing file paths for train, validation, and test data.

    Returns:
        train_med, eval_med, test_med: Lists containing raw training, validation, and test data.
    """
    train_med = [json.loads(line) for line in open('../data/raw/' + data_args.train_file, 'r')]
    eval_med = [json.loads(line) for line in open('../data/raw/' + data_args.validation_file, 'r')]
    test_med = [json.loads(line) for line in open('../data/raw/' + data_args.test_file, 'r')]
    return train_med, eval_med, test_med

def get_clean_data(data, max_samples=None):
    """
    Clean and preprocess data by removing extra whitespaces and creating data_list with sentence pairs and labels.

    Args:
        data: A list of raw data dictionaries containing 'sentence1', 'sentence2', and 'gold_label' keys.
        max_samples (Optional[int]): The maximum number of samples to return from the data list. If not specified, all samples are returned.

    Returns:
        data_list: A list of dictionaries containing 'sentence1', 'sentence2', and 'label' keys.
    """
    s1 = [item['sentence1'].strip() for item in data] # .strip() removes whitespace from a string (e.g, spaces, tabs)
    s2 = [item['sentence2'].strip() for item in data]
    labels = [item['gold_label'] for item in data]

    data_list = [{"sentence1": s1[i], "sentence2": s2[i], "label": labels[i]} for i in range(len(s1))]

    if max_samples is not None:
        max_samples = min(len(data_list), max_samples)
        indices = list(range(len(data_list)))
        random.shuffle(indices)
        data_list = [data_list[i] for i in indices[:max_samples]]

    return data_list

def load_and_clean_data(data_args):
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

    train_med, eval_med, test_med = load_raw_data(data_args)

    train_list = get_clean_data(train_med, data_args.max_train_samples)
    logger.info(f"Loading training data with sample size: {len(train_list)}.")

    eval_list = get_clean_data(eval_med, data_args.max_eval_samples)
    logger.info(f"Loading evaluation data with sample size: {len(eval_list)}.")

    test_list = get_clean_data(test_med, data_args.max_test_samples)
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

def export_clean_data(train_list, eval_list, test_list):
    """
    Export cleaned and preprocessed train, validation, and test data to JSON files.

    Args:
        train_list: A list of cleaned and preprocessed training data.
        eval_list: A list of cleaned and preprocessed evaluation data.
        test_list: A list of cleaned and preprocessed test data.
    """
    export_json(train_list, "../data/clean/clean_train_med.json")
    export_json(eval_list, "../data/clean/clean_eval_med.json")
    export_json(test_list, "../data/clean/clean_test_med.json")

    logger.info(f"Finished exporting cleaned data as json files.")

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    # takes command-line arguments and parses them into dataclasses -> data_args, model_args, training_args
    parser = HfArgumentParser((DataLoadingArguments, MyTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    set_logger_and_verbosity(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load and clean data
    train_list, eval_list, test_list = load_and_clean_data(data_args)

    # export data as json files
    export_clean_data(train_list, eval_list, test_list) 

def _mp_fn(index): # used in conjunction with the below code when running on TPUs, takes in `index` argument which is the process index used by `xla_spawn()`
    # For xla_spawn (TPUs)
    main()
# e.g. for using TPUs
# import torch_xla.distributed.xla_multiprocessing as xmp
# if __name__ == '__main__':
#   xmp.spawn(_mp_fn, args=(), nprocs=8)

if __name__ == "__main__": # if the script is run from the command line, sets __name__ to "__main__" and main() will be executed. if the script is imported into another script, `__name__` will be set to `run_glue` and main() will not be executed
    main()
