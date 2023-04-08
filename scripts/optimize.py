import logging # logs msgs/errors 
import os # helps interact with operating system (e.g. read/write files, navigating files systems)
import sys # helps interact with operating system
# from dataclasses import dataclass, field # helps create functions that store data 
# from typing import Optional, List, Tuple # Optional helps to specify that the param can also be None

# import datasets # can load datasets from hugging face
# import numpy as np
# from datasets import load_dataset # load dataset in -> https://github.com/huggingface/datasets/blob/main/src/datasets/load.py

import transformers
from transformers import (
#     AutoConfig, # auto loads a pre-trained model config (e.g. # of layers) from huggingface
#     AutoModelForSequenceClassification, # auto loads a pre-trained model from huggingface for sequence classification
#     AutoTokenizer, # auto loads a tokenizer from huggingface (tokenizers convert clean text into a format that you can input into a model)
#     DataCollatorWithPadding, # collates samples into batches and does padding
#     EvalPrediction, # for each input, gives predicted scores for each output class
#     EarlyStoppingCallback, # import early stopping
#     HfArgumentParser, # a sub-class of `argparse.ArgumentParser`, helps configure arguments to be passed to command-line
#     PretrainedConfig, # is the base class for pre-trained models, provides common configs/attributes/methods
#     Trainer, # does a lot of the training and evaluation
#     TrainingArguments, # pass this in to an instance of Trainer for argument configurations
#     default_data_collator, # pass this to DataLoader() to collate input data and do padding
     set_seed, # setting a random seed
)
# from transformers.trainer_utils import get_last_checkpoint # returns the path to the last saved checkpoint file during training (useful if training is interrupted)
from transformers.utils import check_min_version # checks if transformers package meets minimum version requirements

# import optuna
# import json

from utils.args_utils import *
from utils.logging_utils import *
from utils.checkpoint_utils import *
from utils.data_load_and_clean_utils import *
from utils.data_preprocessing_utils import *
from utils.hyperparam_utils import *
from utils.model_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

# Get the current working directory
cwd = os.getcwd()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)

# @dataclass
# class DataTrainingArguments:
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
#             train_extension = self.train_file.split(".")[-1] # gets the document type e.g. "csv" or "json". makes sure that it is
#             assert train_extension in ["csv", "json"], "`train_file` should be a csv or json file."
#             validation_extension = self.validation_file.split(".")[-1] # e.g. "csv" or "json". makes sure that it is
#             assert validation_extension == train_extension, "`validation_file` should have the same extension as `train_file`."
#             test_extension = self.test_file.split(".")[-1] # e.g. "csv" or "json"
#             assert test_extension == train_extension, "`test_file` should have the same extension as the `train_file`."

# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """

#     model_name_or_path: str = field( # path to pre-trained model
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )
#     config_name: Optional[str] = field( # path/name to config of model if not the same as model_name
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field( # path to pre-trained tokenizer if not the same as model_name
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field( # specify where pretrained models downloaded from huggingface will be stored, if None, it will be cached in the default directory
#         default=None,
#         metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
#     )
#     use_fast_tokenizer: bool = field( # if True, will use a fast tokenizer from tokenizers library
#         default=True,
#         metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
#     )
#     model_revision: str = field( # specific version of model to use e.g. for `bert-base-uncased` you would use `abcdefg1234567`
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     use_auth_token: bool = field( # set to True to access private models, set to False to only access public models
#         default=False,
#         metadata={
#             "help": (
#                 "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
#                 "with private models)."
#             )
#         },
#     )
#     ignore_mismatched_sizes: bool = field( # set to False to raise an error when head dimensions are different (between pretrained model and current model)
#         default=False,
#         metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
#     )

# @dataclass
# class MyTrainingArguments(TrainingArguments):
#     """
#     Arguments for training/optimizing data in addition to the default params in TrainingArguments() from transformers.
#     """

#     seed: int = field(
#         default = 123,
#         metadata = {"help": "the seed to set"}
#     )
#     n_trials: int = field(
#         default = 100,
#         metadata = {"help": "the number of times to train and evaluate the model during hyperparameter optimization"}
#     )
#     train_subset_perc: float = field(
#         default = 0.2,
#         metadata = {"help": "the proportion of the max_train_samples of the full training data that you want to subset for hyperparameter optimization"}
#     )
#     eval_subset_perc: float = field(
#         default = 0.2,
#         metadata = {"help": "the proportion of the max_eval_samples of the full validation data that you want to subset for hyperparameter optimization"}
#     ) 
#     use_mps_device: bool = field(
#         default = True,
#         metadata = {"help": "whether or not to use mps device"}
#     )

# def parse_arguments():
#     """
#     Parses command line arguments into data classes for data, model, and training.

#     Returns:
#         tuple: A tuple containing data_args, model_args, and training_args.
#     """
#     parser = HfArgumentParser((DataTrainingArguments, ModelArguments, MyTrainingArguments))
#     data_args, model_args, training_args = parser.parse_args_into_dataclasses()
#     return data_args, model_args, training_args

# def setup_logging(training_args):
#     """
#     Configures logging settings for the script.

#     Args:
#         training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.

#     Returns:
#         logger (logging.Logger): A logger object for logging messages.
#     """
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

#     transformers.utils.logging.set_verbosity_info()
#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     logger.warning(
#         f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#         + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
#     )
#     logger.info(f"Training/evaluation parameters {training_args}")

#     return logger

# def detect_last_checkpoint(training_args):
#     """
#     Detects the last checkpoint if available in the output directory.

#     Args:
#         training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.

#     Returns:
#         str or None: The last checkpoint path if available, else None.
#     """
#     last_checkpoint = None
#     if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
#         last_checkpoint = get_last_checkpoint(training_args.output_dir)
#         if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
#             raise ValueError(
#                 f"Output directory ({training_args.output_dir}) already exists and is not empty. "
#                 "Use --overwrite_output_dir to overcome."
#             )
#         elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
#             logger.info(
#                 f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
#                 "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
#             )
#     return last_checkpoint

# def get_data_files(data_args):
#     """
#     Gets the data file paths for train, validation, and test sets.

#     Args:
#         data_args (DataTrainingArguments): An instance of DataTrainingArguments containing the data arguments.

#     Returns:
#         dict: A dictionary containing the file paths for train, validation, and test sets.
#     """
#     data_files = {
#         "train": "../data/clean/" + data_args.train_file,
#         "validation": "../data/clean/" + data_args.validation_file,
#         "test": "../data/clean/" + data_args.test_file,
#     }
#     return data_files

# def log_data_files(logger, data_files):
#     """
#     Logs the data file paths using the logger.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         data_files (dict): A dictionary containing the file paths for train, validation, and test sets.
#     """
#     for key in data_files.keys():
#         logger.info(f"load a local file for {key}: {data_files[key]}")


# def load_clean_datasets(logger, data_files, model_args, data_args):
#     """
#     Loads clean datasets from the provided data files.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         data_files (dict): A dictionary containing the file paths for train, validation, and test sets.
#         model_args (ModelArguments): An instance of ModelArguments containing the model arguments.
#         data_args (DataTrainingArguments): An instance of DataTrainingArguments containing the data arguments.

#     Returns:
#         datasets.DatasetDict: A dictionary containing the clean datasets for train, validation, and test sets.
#     """
#     logger.info(f"Loading clean datasets...")

#     if data_args.train_file.endswith(".csv"):
#         clean_datasets = load_dataset(
#             "csv",
#             data_files=data_files,
#             cache_dir=model_args.cache_dir,
#             use_auth_token=True if model_args.use_auth_token else None,
#         )
#     else:
#         clean_datasets = load_dataset(
#             "json",
#             data_files=data_files,
#             cache_dir=model_args.cache_dir,
#             use_auth_token=True if model_args.use_auth_token else None,
#         )
#     return clean_datasets

# def create_study(logger):
#     """
#     Creates an Optuna study for hyperparameter optimization.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.

#     Returns:
#         optuna.study.Study: An Optuna study object for hyperparameter optimization.
#     """
#     logger.info(f"Creating study...")

#     study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), study_name="Optimize Hyperparameters")
#     study.enqueue_trial({"learning_rate": 3e-5, "per_device_train_batch_size": 8, "num_train_epochs": 3, "max_seq_length": 128, "weight_decay": 1e-2})
#     return study

# def get_search_space(param_name):
#     """
#     Defines the search space for hyperparameter optimization.

#     Args:
#         param_name (str): The name of the hyperparameter for which to get the search space.

#     Returns:
#         tuple: A tuple containing the search space for the specified hyperparameter.
#     """
#     search_space = {
#         "learning_rate": (2e-5, 5e-5),
#         "per_device_train_batch_size": (8, 16),
#         "num_train_epochs": (2, 4),
#         "max_seq_length": (128, 384),
#         "weight_decay": (1e-4, 1e-2)
#     }
#     return search_space[param_name]

# def get_hyperparameters(trial, get_search_space, logger):
#     """
#     Gets the hyperparameters to try from the search space and logs the current trial number and hyperparameter values.

#     Args:
#         trial (optuna.trial.Trial): An Optuna trial object for the current trial.
#         get_search_space (function): A function to get the search space for a specific hyperparameter.
#         logger (logging.Logger): A logger object for logging messages.

#     Returns:
#         tuple: A tuple containing the suggested hyperparameter values.
#     """
#     learning_rate = trial.suggest_loguniform("learning_rate", *get_search_space("learning_rate")) # e.g. trial.suggest_loguniform() accesses the trainer object and modifies its learning rate
#     per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", *get_search_space("per_device_train_batch_size"))
#     num_train_epochs = trial.suggest_int("num_train_epochs", *get_search_space("num_train_epochs"))
#     max_seq_length = trial.suggest_int("max_seq_length", *get_search_space("max_seq_length"), step=64)
#     weight_decay = trial.suggest_loguniform("weight_decay", *get_search_space("weight_decay"))
    
#     logger.info(f"Starting trial {trial.number} with learning_rate = {learning_rate:.5f}, per_device_train_batch_size = {per_device_train_batch_size}, num_train_epochs = {num_train_epochs}, max_seq_length = {max_seq_length}, weight_decay = {weight_decay:.5f}.")
    
#     return learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay

# def update_training_args_with_hyperparams_and_early_stopping(logger, training_args, learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay):
#     """
#     Sets the training arguments to the suggested hyperparameter values and configures early stopping.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.
#         learning_rate (float): The suggested learning rate value.
#         per_device_train_batch_size (int): The suggested per-device train batch size value.
#         num_train_epochs (int): The suggested number of training epochs.
#         max_seq_length (int): The suggested maximum sequence length value.
#         weight_decay (float): The suggested weight decay value.

#     Returns:
#         MyTrainingArguments: The updated instance of MyTrainingArguments with the suggested hyperparameter values.
#     """
#     logger.info(f"Setting training_args attributes to suggested values.")
    
#     # set attributes for early stopping patience
#     setattr(training_args, "load_best_model_at_end", True)
#     setattr(training_args, "evaluation_strategy", "steps")
#     setattr(training_args, "save_strategy", "steps")
#     setattr(training_args, "metric_for_best_model", "eval_loss")
#     setattr(training_args, "eval_steps", 200)

#     # feed the suggested values into training_args
#     setattr(training_args, "learning_rate", learning_rate)
#     setattr(training_args, "per_device_train_batch_size", per_device_train_batch_size)
#     setattr(training_args, "num_train_epochs", num_train_epochs)
#     setattr(training_args, "max_seq_length", max_seq_length)
#     setattr(training_args, "weight_decay", weight_decay)
#     return training_args

# def get_label_information(logger, clean_datasets: datasets.DatasetDict) -> Tuple[List[str], int]:
#     """
#     Gets label information, including the unique labels and the number of labels.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         clean_datasets (datasets.DatasetDict): A dictionary containing the clean datasets for train, validation, and test sets.

#     Returns:
#         tuple: A tuple containing a sorted list of unique labels and the number of labels.
#     """
#     logger.info(f"Getting label information...")

#     label_list = clean_datasets["train"].unique("label")
#     label_list.sort()
#     num_labels = len(label_list)
#     return label_list, num_labels

# def load_config_tokenizer_model(logger, model_args, num_labels):
#     """
#     Loads the pre-trained configuration, tokenizer, and model.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         model_args (ModelArguments): An instance of ModelArguments containing the model arguments.
#         num_labels (int): The number of unique labels in the dataset.

#     Returns:
#         tuple: A tuple containing the configuration, tokenizer, and model objects.
#     """
#     logger.info(f"Loading pre-trained config, tokenizer, and model...")

#     config = AutoConfig.from_pretrained(
#         model_args.config_name if model_args.config_name else model_args.model_name_or_path,
#         num_labels=num_labels,
#         finetuning_task="text-classification",
#         cache_dir=model_args.cache_dir,
#         revision=model_args.model_revision,
#         use_auth_token=True if model_args.use_auth_token else None,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
#         cache_dir=model_args.cache_dir,
#         use_fast=model_args.use_fast_tokenizer,
#         revision=model_args.model_revision,
#         use_auth_token=True if model_args.use_auth_token else None,
#     )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_args.model_name_or_path,
#         from_tf=bool(".ckpt" in model_args.model_name_or_path),
#         config=config,
#         cache_dir=model_args.cache_dir,
#         revision=model_args.model_revision,
#         use_auth_token=True if model_args.use_auth_token else None,
#         ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
#     )
#     return config, tokenizer, model

# def get_max_seq_length(max_seq_length, tokenizer, logger):
#     """
#     Args:
#         max_seq_length (int): the max allowed length for the tokenizer
#         tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): the tokenizer
#         logger (logging.Logger): the logger

#     Returns:
#         max_seq_length (int): the max allowed length for the tokenizer
#     """
#     logger.info(f"Getting max_seq_length...")

#     if max_seq_length > tokenizer.model_max_length: # if max_seq_length is greater than the max allowed length for the tokenizer, gives a warning... 
#         logger.warning(
#             f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
#             f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
#         )
#         max_seq_length = min(max_seq_length, tokenizer.model_max_length) # ...and sets the max_seq_length to the max allowed length for tokenizer
#     return max_seq_length 
 
# def get_non_label_column_names(logger, clean_datasets):
#     """
#     Gets the column names in the dataset that are not 'label'.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         clean_datasets (datasets.DatasetDict): A dictionary containing the clean datasets for train, validation, and test sets.

#     Returns:
#         list: A list of column names that are not 'label'.
#     """
#     logger.info(f"Getting non_label_column_names...")

#     non_label_column_names = [name for name in clean_datasets["train"].column_names if name != "label"]
#     return non_label_column_names

# def get_sentence_keys(logger, non_label_column_names):
#     """
#     Gets the sentence1_key and sentence2_key based on the non_label_column_names.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         non_label_column_names (list): A list of column names that are not 'label'.

#     Returns:
#         tuple: A tuple containing the sentence1_key and sentence2_key.
#     """
#     logger.info(f"Getting sentence1_key, sentence2_key...")

#     if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
#         sentence1_key, sentence2_key = "sentence1", "sentence2"
#     else:
#         if len(non_label_column_names) >= 2:
#             sentence1_key, sentence2_key = non_label_column_names[:2]
#         else:
#             sentence1_key, sentence2_key = non_label_column_names[0], None
#     return sentence1_key, sentence2_key

# def get_label_to_id(logger, config, num_labels, label_list):
#     """
#     Gets the mapping of labels to IDs.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         config (PretrainedConfig): The pre-trained model configuration.
#         num_labels (int): The number of unique labels in the dataset.
#         label_list (list): A list of unique labels.

#     Returns:
#         dict: A dictionary containing the mapping of labels to IDs.
#     """
#     logger.info(f"Getting label_to_id...")

#     # this function is necessary because the label IDs are used internally by the model during training, but the label names are used externally to interpret the model's predictions.
#     # it makes sure that the internal representations match the external representations
#     label_to_id = None
#     if (config.label2id != PretrainedConfig(num_labels=num_labels).label2id):
#         label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
#         if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
#             label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
#         else:
#             logger.warning(
#                 "Your model seems to have been trained with labels, but they don't match the dataset: ",
#                 f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
#                 "\nIgnoring the model labels as a result.",
#             )
#     label_to_id = {v: i for i, v in enumerate(label_list)}
#     return label_to_id

# def preprocess_clean_datasets(logger, clean_datasets, config, num_labels, label_list):
#     """
#     Preprocesses the clean datasets and gets the sentence keys and label_to_id mapping.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         clean_datasets (datasets.DatasetDict): A dictionary containing the clean datasets for train, validation, and test sets.
#         config (PretrainedConfig): The pre-trained model configuration.
#         num_labels (int): The number of unique labels in the dataset.
#         label_list (list): A list of unique labels.

#     Returns:
#         tuple: A tuple containing the sentence1_key, sentence2_key, and label_to_id mapping.
#     """
#     logger.info(f"Preprocessing clean datasets...")

#     non_label_column_names = get_non_label_column_names(logger, clean_datasets)
#     sentence1_key, sentence2_key = get_sentence_keys(logger, non_label_column_names)
#     label_to_id = get_label_to_id(logger, config, num_labels, label_list)
#     return sentence1_key, sentence2_key, label_to_id

# # preprocesses clean data into tokens to input into the model
# def preprocess_tokenize_function(example, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer): 
#     """
#     Preprocesses and tokenizes a single example.

#     Args:
#         example (dict): A single example from the dataset.
#         max_seq_length (int): The maximum sequence length allowed.
#         sentence1_key (str): The key for the first sentence in the example.
#         sentence2_key (str): The key for the second sentence in the example, if applicable.
#         label_to_id (dict): A dictionary containing the mapping of labels to IDs.
#         tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.

#     Returns:
#         dict: A dictionary containing the tokenized input IDs, attention masks, token type IDs, and label IDs.
#     """
#     # Tokenize the texts
#     args = (
#         (example[sentence1_key],) if sentence2_key is None else (example[sentence1_key], example[sentence2_key]) 
#     )
#     result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True) # tokenizer() takes input example sentence1 and sentence2 and -> {'input_ids':[101, 1996, 4248, 2829, 4419, 0, 0], 'attention_mask':[1,1,1,1,1,0,0], 'token_type_ids':[0,0,0,1,1,0,0]}
#     # e.g. example[sentence1_key] = ['The cat in the hat.', 'The fox in the shoe.', 'The dwarf in the mug.'] (this is a list of sentence1's for a given batch)
#     # e.g. example[sentence2_key] = ['The dog in the mut.', 'The woof in the bark.', 'The giant in the cave.'] (this is a list of sentence2's for a given batch)

#     # Map labels to IDs 
#     if label_to_id is not None and "label" in example:
#         result["label"] = [(label_to_id[l] if l != -1 else -1) for l in example["label"]] # e.g. for l in example["label"] = [0,1,2,0,2,1,1]. checks that the label is not unknown (-1 represents unknown or NaN)
#     return result # e.g. results for a batch of 2 sentence pair examples {'input_ids':[[101, 1996, 4248, 2829, 4419, 0, 0], [92, 903, 384, 2700, 0, 0, 0]]'attention_mask':[[1,1,1,1,1,0,0],[1,1,1,1,0,0,0]], 'token_type_ids':[[0,0,0,1,1,0,0],[0,0,1,1,0,0,0]], 'label':[1,0]}

# def preprocess_datasets(logger, clean_datasets, training_args, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer): # don't need to feed in example because it's a built in variable name used in the `map` method of hugging face datasets library used to represent an individual example of `clean_datasets`, which is a datasets library object
#     """
#     Applies the preprocess_tokenize_function to all examples in the clean datasets.

#     Args:
#         logger (logging.Logger): A logger object for logging messages.
#         clean_datasets (datasets.DatasetDict): A dictionary containing the clean datasets for train, validation, and test sets.
#         training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.
#         max_seq_length (int): The maximum sequence length allowed.
#         sentence1_key (str): The key for the first sentence in the examples.
#         sentence2_key (str): The key for the second sentence in the examples, if applicable.
#         label_to_id (dict): A dictionary containing the mapping of labels to IDs.
#         tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.

#     Returns:
#         datasets.DatasetDict: The preprocessed and tokenized datasets for train, validation, and test sets.
#     """
#     logger.info(f"Applying preprocess_tokenize_function to examples in clean datasets...")

#     with training_args.main_process_first(desc="dataset map pre-processing"):
#         clean_datasets = clean_datasets.map( # map applies a function to all the examples in a dataset
#             lambda example: preprocess_tokenize_function(example, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer), # takes in an example, and applies the preprocess function to it
#             batched=True,
#             load_from_cache_file=True,
#             desc="Running tokenizer on dataset",
#         )
#     return clean_datasets

# def get_train_subsets(clean_datasets, data_args, training_args):
#     """
#     Extracts a subset of the training dataset based on the training_args.train_subset_perc parameter.

#     Args:
#         clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
#         data_args (DataArguments): The data arguments containing dataset-related configurations.
#         training_args (TrainingArguments): The training arguments containing training-related configurations.

#     Returns:
#         datasets.Dataset: The subset of the training dataset.
#     """
#     train_dataset = clean_datasets["train"]
#     if data_args.max_train_samples is not None:
#         max_train_samples = min(len(train_dataset), data_args.max_train_samples)
#         train_dataset = train_dataset.select(range(max_train_samples))

#     short_train = train_dataset.select(range(int(training_args.train_subset_perc * len(train_dataset))))
#     return short_train

# def get_eval_subsets(clean_datasets, data_args, training_args):
#     """
#     Extracts a subset of the evaluation dataset based on the training_args.eval_subset_perc parameter.

#     Args:
#         clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
#         data_args (DataArguments): The data arguments containing dataset-related configurations.
#         training_args (TrainingArguments): The training arguments containing training-related configurations.

#     Returns:
#         datasets.Dataset: The subset of the evaluation dataset.
#     """
#     eval_dataset = clean_datasets["validation"]
#     if data_args.max_eval_samples is not None:
#         max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
#         eval_dataset = eval_dataset.select(range(max_eval_samples))

#     short_eval = eval_dataset.select(range(int(training_args.eval_subset_perc * len(eval_dataset))))
#     return short_eval

# def get_subsets(logger, clean_datasets, data_args, training_args):
#     """
#     Gets the subsets of train and eval datasets based on the specified training and evaluation percentage parameters.

#     Args:
#         logger (logging.Logger): The logger to display information during the process.
#         clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
#         data_args (DataArguments): The data arguments containing dataset-related configurations.
#         training_args (TrainingArguments): The training arguments containing training-related configurations.

#     Returns:
#         Tuple[datasets.Dataset, datasets.Dataset]: The subsets of the training and evaluation datasets.
#     """
#     logger.info(f"Getting subsets...")

#     short_train = get_train_subsets(clean_datasets, data_args, training_args)
#     short_eval = get_eval_subsets(clean_datasets, data_args, training_args)
#     return short_train, short_eval

# def create_data_collator(logger, training_args, tokenizer):
#     """
#     Creates a data collator based on the training arguments and tokenizer.

#     Args:
#         logger (logging.Logger): The logger to display information during the process.
#         training_args (TrainingArguments): The training arguments containing training-related configurations.
#         tokenizer (PreTrainedTokenizer): The tokenizer to be used in the data collator.

#     Returns:
#         DataCollator: The data collator to be used during training and evaluation.
#     """
#     logger.info(f"Creating data_collator...")

#     data_collator = default_data_collator
#     if training_args.fp16:
#         data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
#     return data_collator

# def compute_metrics(p: EvalPrediction): # p or EvalPrediction is a tuple with `predictions`: predicted probs of the model and `label_ids`: true labels
#     """
#     Computes the accuracy of the predictions against the true labels.

#     Args:
#         p (EvalPrediction): A named tuple containing two fields: predictions and label_ids.

#     Returns:
#         Dict[str, float]: A dictionary with a single key-value pair representing the accuracy.
#     """
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # isinstance() checks if it is a tuple, this line extracts the model predicted probs
#     preds = np.argmax(preds, axis=1) # change pred format to be suitable for computing evaluation metrics
#     # np.argmax() gets the index of the highest probability class label predicted by model
#     return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()} # gets an array of True/False, .astype(np.float32) converts this to 1 or 0, then take the mean for accuracy

# def train_and_evaluate(trial, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets):
#     """
#     Trains and evaluates the model with the given hyperparameters and datasets.

#     Args:
#         trial (optuna.trial.Trial): An Optuna trial object that contains trial information and hyperparameters.
#         data_args (DataArguments): The data arguments containing dataset-related configurations.
#         model_args (ModelArguments): The model arguments containing model-related configurations.
#         training_args (TrainingArguments): The training arguments containing training-related configurations.
#         logger (logging.Logger): The logger to display information during the process.
#         last_checkpoint (str): The path to the last checkpoint from which to resume training.
#         clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.

#     Returns:
#         float: The accuracy of the model on the evaluation dataset.
#     """
    
#     # get hyperparameters for the current trial
#     learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay = get_hyperparameters(trial, get_search_space, logger)
    
#     # update training arguments with the trial hyperparameters
#     training_args = update_training_args_with_hyperparams_and_early_stopping(logger, training_args, learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay)

#     # get label information from the dataset
#     label_list, num_labels = get_label_information(logger, clean_datasets)

#     # load pre-trained config, tokenizer, and model
#     config, tokenizer, model = load_config_tokenizer_model(logger, model_args, num_labels)
    
#     # update max_seq_length if it is too long for the tokenizer
#     max_seq_length = get_max_seq_length(max_seq_length, tokenizer, logger)

#     # preprocess datasets to get sentence1_key, sentence2_key, and label_to_id
#     sentence1_key, sentence2_key, label_to_id = preprocess_clean_datasets(logger, clean_datasets, config, num_labels, label_list)
    
#     # preprocess datasets by tokenizing and truncating
#     clean_datasets = preprocess_datasets(logger, clean_datasets, training_args, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer)
    
#     # get train and eval subsets
#     short_train, short_eval = get_subsets(logger, clean_datasets, data_args, training_args)

#     # create data collator for padding and batching
#     data_collator = create_data_collator(logger, training_args, tokenizer)
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=short_train,
#         eval_dataset=short_eval,
#         compute_metrics=compute_metrics,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
#     )
    
#     logger.info(f"*** Training Trial {trial.number} ***")
    
#     # determine the checkpoint to resume training from
#     checkpoint = last_checkpoint if last_checkpoint is not None else training_args.resume_from_checkpoint

#     # train the model (fine-tune)
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
#     # get training metrics
#     metrics = train_result.metrics
#     # store the number of training samples
#     metrics["train_samples"] = len(short_train)

#     logger.info(f"*** Evaluating Trial {trial.number} ***")

#     # evaluate the model
#     metrics = trainer.evaluate(eval_dataset=short_eval)
#     # store the number of evaluation samples
#     metrics["eval_samples"] = len(short_eval)

#     logger.info(f"Trial {trial.number} finished with accuracy: {metrics['eval_accuracy']:.4f}.")

#     return metrics["eval_accuracy"]

# # define the hyperparameter optimization function
# def objective(trial, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets):
#     """
#     The objective function for the hyperparameter optimization process.

#     Args:
#         trial (optuna.trial.Trial): An Optuna trial object that contains trial information and hyperparameters.
#         data_args (DataArguments): The data arguments containing dataset-related configurations.
#         model_args (ModelArguments): The model arguments containing model-related configurations.
#         training_args (TrainingArguments): The training arguments containing training-related configurations.
#         logger (logging.Logger): The logger to display information during the process.
#         last_checkpoint (str): The path to the last checkpoint from which to resume training.
#         clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.

#     Returns:
#         float: The accuracy of the model on the evaluation dataset.
#     """
#     return train_and_evaluate(
#             trial=trial,
#             data_args=data_args,
#             model_args=model_args,
#             training_args=training_args,
#             logger=logger,
#             last_checkpoint=last_checkpoint,
#             clean_datasets=clean_datasets
#         )

# def optimize_hyperparameters(study, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets):
#     """
#     Optimizes the hyperparameters using the objective function and Optuna.

#     Args:
#         study (optuna.study.Study): An Optuna study object to manage the optimization process.
#         data_args (DataArguments): The data arguments containing dataset-related configurations.
#         model_args (ModelArguments): The model arguments containing model-related configurations.
#         training_args (TrainingArguments): The training arguments containing training-related configurations.
#         logger (logging.Logger): The logger to display information during the process.
#         last_checkpoint (str): The path to the last checkpoint from which to resume training.
#         clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
#     """
#     study.optimize(lambda trial: objective(trial, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets),
#                    n_trials=training_args.n_trials)
#     return study

# def log_best_hyperparameters(study, logger):
#     """
#     Logs the best hyperparameters found during the optimization process.

#     Args:
#         study (optuna.study.Study): An Optuna study object to manage the optimization process.
#         logger (logging.Logger): The logger to display information during the process.
#     """
#     best_param_str = ", ".join(
#         f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
#         for k, v in study.best_params.items()
#     )
#     logger.info(f"Hyperparameter optimization completed with best hyperparameters: {best_param_str} with an accuracy of {study.best_value:.4f} during the trial evaluation.")

# def save_optimized_hyperparams(logger, study, output_path):
#     """
#     Saves the optimized hyperparameters as a JSON file.

#     Args:
#         logger (logging.Logger): The logger to display information during the process.
#         study (optuna.study.Study): An Optuna study object to manage the optimization process.
#         output_path (str): The path to the output file where the optimized hyperparameters will be saved.
#     """
#     logger.info(f"Saving optimized hyperparameters...")

#     optimized_hyperparams = {
#         'best_hyperparameters': study.best_params,
#         'best_accuracy': study.best_value
#     }
#     with open(output_path, 'w') as f:
#         json.dump(optimized_hyperparams, f)

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

    # Create Optuna study
    study = create_study(logger)

    # Optimize hyperparameters
    study = optimize_hyperparameters(study, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets)

    # Log best hyperparameters
    log_best_hyperparameters(study, logger)

    # Save optimized hyperparameters
    output_path = '../results/hyperparameter_optimization/optimized_hyperparameters.json'
    save_optimized_hyperparams(logger, study, output_path)

def _mp_fn(index): # used in conjunction with the below code when running on TPUs, takes in `index` argument which is the process index used by `xla_spawn()`
    # For xla_spawn (TPUs)
    main()
# e.g. for using TPUs
# import torch_xla.distributed.xla_multiprocessing as xmp
# if __name__ == '__main__':
#   xmp.spawn(_mp_fn, args=(), nprocs=8)

if __name__ == "__main__": # if the script is run from the command line, sets __name__ to "__main__" and main() will be executed. if the script is imported into another script, `__name__` will be set to `run_glue` and main() will not be executed
    main()