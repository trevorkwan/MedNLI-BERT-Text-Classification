from dataclasses import dataclass, field # helps create functions that store data 
import os
import sys # helps interact with operating system
import logging # logs msgs/errors 

from transformers import set_seed, TrainingArguments
from transformers.utils import check_min_version # checks if transformers package meets minimum version requirements
from typing import Optional # Optional helps to specify that the param can also be None

from utils.args_utils import parse_arguments
from utils.logging_utils import set_logger_and_verbosity
from utils.data_load_and_clean_utils import load_and_clean_data, export_clean_data

# @dataclass
# class DataLoadingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model with for training and evaluation.
#     """

#     train_file: Optional[str] = field( # specify path to training data
#         default=None, metadata={"help": "A csv or a json file containing the training data."}
#     )
#     validation_file: Optional[str] = field( # specify path to validation data
#         default=None, metadata={"help": "A csv or a json file containing the validation data."}
#     )
#     test_file: Optional[str] = field( # specify path to testing data
#         default=None, metadata={"help": "A csv or a json file containing the test data."}
#     )
#     max_train_samples: Optional[int] = field( # used for quicker training
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncates the number of training examples to this"
#             )
#         },
#     )
#     max_eval_samples: Optional[int] = field( # for quicker validation
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of evaluation examples to this"
#             )
#         },
#     )
#     max_test_samples: Optional[int] = field( # for quicker testing
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of predictions to this."
#             )
#         },
#     )

#     def __post_init__(self): # does additional checks after defining params
#         if self.train_file is None or self.validation_file is None or self.test_file is None:
#             raise ValueError("Need a training/validation/test file.")
#         else:
#             train_extension = self.train_file.split(".")[-1] # gets the document type e.g. "jsonl", makes sure that it is
#             assert train_extension in ["jsonl"], "`train_file` should be a jsonl file."
#             validation_extension = self.validation_file.split(".")[-1] # e.g. "jsonl". makes sure that it is
#             assert validation_extension == train_extension, "`validation_file` should have the same extension as `train_file`."

# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """
#     pass

# @dataclass
# class MyTrainingArguments(TrainingArguments):
#     """
#     My training arguments
#     """

#     seed: int = field(
#         default = 123,
#         metadata = {"help": "the seed to set"}
#     )

#     output_dir: str = field(
#         default=None,
#         metadata={"help": "specify output directory for parser"}
#     )
    
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

# Get the current working directory
cwd = os.getcwd()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)

def main():

    # Parse arguments
    data_args, model_args, training_args = parse_arguments()

    # Setup logging
    logger = logging.getLogger(__name__) # creates a logger object with the "name" get_best_model, to identify where these logger messages are coming from
    set_logger_and_verbosity(logger)

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
