from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
from transformers.training_args import TrainingArguments


@dataclass
class DataLoadingArguments:
    """
    Arguments pertaining to the input data used for training, evaluation, and testing the model.
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
    sentence1_key: Optional[str] = field( # specify the key for the first sentence in the data
        default="sentence1",
        metadata={"help": "The key in the json files corresponding to the premise sentence."},
    )
    sentence2_key: Optional[str] = field( # specify the key for the second sentence in the data
        default="sentence2",
        metadata={"help": "The key in the json files corresponding to the hypothesis sentence."},
    )
    label_key: Optional[str] = field( # specify the key for the label in the data
        default="gold_label",
        metadata={"help": "The key in the json files corresponding to the label."},
    )

    def __post_init__(self): # does additional checks after defining params
        if self.train_file is None or self.validation_file is None or self.test_file is None:
            raise ValueError("Need a training/validation/test file.")
        else:
            train_extension = self.train_file.split(".")[-1] # gets the document type e.g. "jsonl", makes sure that it is
            assert train_extension in ["jsonl", "json"], "`train_file` should be a jsonl file or json file."
            validation_extension = self.validation_file.split(".")[-1] # e.g. "jsonl". makes sure that it is
            assert validation_extension == train_extension, "`validation_file` should have the same extension as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to the pre-trained model, configuration, and tokenizer used for fine-tuning.
    """

    model_name_or_path: Optional[str] = field( # path to pre-trained model
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field( # path/name to config of model if not the same as model_name
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field( # path to pre-trained tokenizer if not the same as model_name
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field( # specify where pretrained models downloaded from huggingface will be stored, if None, it will be cached in the default directory
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field( # if True, will use a fast tokenizer from tokenizers library
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field( # specific version of model to use e.g. for `bert-base-uncased` you would use `abcdefg1234567`
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field( # set to True to access private models, set to False to only access public models
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field( # set to False to raise an error when head dimensions are different (between pretrained model and current model)
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    """
    Custom training arguments that include seed, output directory, hyperparameters, and early stopping configurations.
    """

    seed: int = field(
        default = 123,
        metadata = {"help": "the seed to set"}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "specify output directory for parser"}
    )
    n_trials: int = field(
        default = 100,
        metadata = {"help": "the number of times to train and evaluate the model during hyperparameter optimization"}
    )
    train_subset_perc: float = field(
        default = 0.2,
        metadata = {"help": "the proportion of the max_train_samples of the full training data that you want to subset for hyperparameter optimization"}
    )
    eval_subset_perc: float = field(
        default = 0.2,
        metadata = {"help": "the proportion of the max_eval_samples of the full validation data that you want to subset for hyperparameter optimization"}
    ) 
    use_mps_device: bool = field(
        default = True,
        metadata = {"help": "whether or not to use mps device"}
    )
    learning_rate: float = field(
        default = 3e-5,
        metadata = {"help": "the hyperparameter learning rate to use for training, default is 3e-5"}
    )
    per_device_train_batch_size: int = field(
        default = 8,
        metadata = {"help": "the hyperparameter batch size to use for training, default is 8"}
    )
    num_train_epochs: int = field(
        default = 3,
        metadata = {"help": "the hyperparameter number of epochs to use for training, default is 3"}
    )
    max_seq_length: int = field(
        default = 128,
        metadata = {"help": "the hyperparameter max sequence length to use for training, default is 128"}
    )
    weight_decay: float = field(
        default = 1e-2,
        metadata = {"help": "the hyperparameter weight decay to use for training, default is 1e-2"}
    )
    use_optimized_hyperparams: bool = field(
        default = True,
        metadata = {
        "help": "set to False if the optimized_hyperparameters file exists but you don't want to use ANY of the optimized hyperparameters."
                    "default is True. if True, will use the optimized hyperparameters instead of default values for the hyperparameters that are not user specified."
                    }
    )
    load_best_model_at_end: bool = field(
        default = True,
        metadata = {"help": "for early stopping, whether or not to load the best model at the end of training, default is True"}
    )
    evaluation_strategy: str = field(
        default = "steps",
        metadata = {"help": "for early stopping, the hyperparameter evaluation strategy to use for training, default is 'steps'"}
    )
    save_strategy: str = field(
        default = "steps",
        metadata = {"help": "for early stopping, the hyperparameter save strategy to use for training, default is 'steps'"}
    )
    metric_for_best_model: str = field(
        default = "eval_loss",
        metadata = {"help": "for early stopping, the hyperparameter metric to use for training, default is 'eval_loss'"}
    )
    num_times_eval_per_epoch: float = field(
        default = 2.0,
        metadata = {"help": "for early stopping, the number of times to evaluate (check for overfitting) per epoch to use for training, default is 2.0"}
    )

def parse_arguments():
    """
    Parses command line arguments and returns data, model, and training argument data classes.

    Returns:
        tuple: A tuple containing data_args, model_args, and training_args as data classes.
    """
    parser = HfArgumentParser((DataLoadingArguments, ModelArguments, MyTrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    return data_args, model_args, training_args
