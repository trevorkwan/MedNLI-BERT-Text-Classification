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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    # takes command-line arguments and parses them into dataclasses -> data_args, model_args, training_args
    parser = HfArgumentParser((DataLoadingArguments, MyTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # log format
        datefmt="%m/%d/%Y %H:%M:%S", # datetime format
        handlers=[logging.StreamHandler(sys.stdout)], # where to send the log messages e.g. a file, the console
    )
    
    transformers.utils.logging.set_verbosity_info() # sets log level to `INFO` or higher
    log_level = training_args.get_process_log_level() # sets log_level to value specified in `training_args` object
    logger.setLevel(log_level) # assigns it to the logger object
    datasets.utils.logging.set_verbosity(log_level) # ensures that log messages emitted by datasets library are `log_level` or higher
    transformers.utils.logging.set_verbosity(log_level) # ensures that log messages emitted by transformers library are `log_level` or higher
    transformers.utils.logging.enable_default_handler() # enables default handler for transformers library
    transformers.utils.logging.enable_explicit_format() # enables a different format for log messages
    
    # Set seed to get the same subset.
    set_seed(training_args.seed)

    # load train + eval + test mednli_data as a list
    logger.info(f"Starting to load data from files: {data_args.train_file, data_args.validation_file, data_args.test_file}") 

    train_med = [json.loads(line) for line in open('../data/raw/' + data_args.train_file, 'r')]
    eval_med = [json.loads(line) for line in open('../data/raw/' + data_args.validation_file, 'r')]
    test_med = [json.loads(line) for line in open('../data/raw/' + data_args.test_file, 'r')]

    # training data extract sentence1, sentence2, and label from list of dictionaries
    train_s1 = [item['sentence1'] for item in train_med]
    train_s1 = [x.strip() for x in train_s1] # .strip() removes whitespace from a string (e.g, spaces, tabs)
    train_s2 = [item['sentence2'] for item in train_med]
    train_s2 = [x.strip() for x in train_s2]
    train_labels = [item['gold_label'] for item in train_med]

    # create train list of dict
    train_list = []
    for i in range(len(train_s1)):
        train_dict = {
            "sentence1": train_s1[i],
            "sentence2": train_s2[i],
            "label": train_labels[i]
        }
        train_list.append(train_dict)
    
    # subset train if needed
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_list), data_args.max_train_samples)
        indices = list(range(len(train_list)))
        random.shuffle(indices)
        train_list = [train_list[i] for i in indices[:max_train_samples]]
    
    logger.info(f"Loading training data with sample size: {len(train_list)}.") 

    # eval data extract sentence1, sentence2, and label from list of dictionaries
    eval_s1 = [item['sentence1'] for item in eval_med]
    eval_s1 = [x.strip() for x in eval_s1]
    eval_s2 = [item['sentence2'] for item in eval_med]
    eval_s2 = [x.strip() for x in eval_s2]
    eval_labels = [item['gold_label'] for item in eval_med]

    # create eval list of dict
    eval_list = []
    for i in range(len(eval_s1)):
        eval_dict = {
            "sentence1": eval_s1[i],
            "sentence2": eval_s2[i],
            "label": eval_labels[i]
        }
        eval_list.append(eval_dict)
    
    # subset eval if needed
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_list), data_args.max_eval_samples)
        indices = list(range(len(eval_list)))
        random.shuffle(indices)
        eval_list = [eval_list[i] for i in indices[:max_eval_samples]]

    logger.info(f"Loading evaluation data with sample size: {len(eval_list)}.") 

    # test data extract sentence1, sentence2, and label from list of dictionaries
    test_s1 = [item['sentence1'] for item in test_med]
    test_s1 = [x.strip() for x in test_s1]
    test_s2 = [item['sentence2'] for item in test_med]
    test_s2 = [x.strip() for x in test_s2]
    test_labels = [item['gold_label'] for item in test_med]

    # create test list of dict
    test_list = []
    for i in range(len(test_s1)):
        test_dict = {
            "sentence1": test_s1[i],
            "sentence2": test_s2[i],
            "label": test_labels[i]
        }
        test_list.append(test_dict)
    
    # subset test if needed
    if data_args.max_test_samples is not None:
        max_test_samples = min(len(test_list), data_args.max_test_samples)
        indices = list(range(len(test_list)))
        random.shuffle(indices)
        test_list = [test_list[i] for i in indices[:max_test_samples]]
    
    logger.info(f"Loading testing data with sample size: {len(test_list)}.") 

    # export train/eval/test as .json
    with open("../data/clean/clean_train_med.json", "w") as outfile:
        json.dump(train_list, outfile)

    with open("../data/clean/clean_eval_med.json", "w") as outfile:
        json.dump(eval_list, outfile)

    with open("../data/clean/clean_test_med.json", "w") as outfile:
        json.dump(test_list, outfile)
    
    logger.info(f"Finished exporting cleaned data as json files.") 

def _mp_fn(index): # used in conjunction with the below code when running on TPUs, takes in `index` argument which is the process index used by `xla_spawn()`
    # For xla_spawn (TPUs)
    main()
# e.g. for using TPUs
# import torch_xla.distributed.xla_multiprocessing as xmp
# if __name__ == '__main__':
#   xmp.spawn(_mp_fn, args=(), nprocs=8)

if __name__ == "__main__": # if the script is run from the command line, sets __name__ to "__main__" and main() will be executed. if the script is imported into another script, `__name__` will be set to `run_glue` and main() will not be executed
    main()