from transformers import HfArgumentParser
from dataclasses import dataclass
from typing import Optional
from transformers.training_args import TrainingArguments


@dataclass
class DataTrainingArguments:
    # Add your data-related arguments here, e.g.: data_dir: str
    pass


@dataclass
class ModelArguments:
    # Add your model-related arguments here, e.g.: model_name_or_path: str
    pass


@dataclass
class MyTrainingArguments(TrainingArguments):
    # Add any custom training arguments here, or override the default ones
    pass

def parse_arguments():
    """
    Parses command line arguments into data classes for data, model, and training.

    Returns:
        tuple: A tuple containing data_args, model_args, and training_args.
    """
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, MyTrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    return data_args, model_args, training_args
