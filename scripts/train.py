import logging # logs msgs/errors 
import os # helps interact with operating system (e.g. read/write files, navigating files systems)
import random # helps generate random numbers
import sys # helps interact with operating system
from dataclasses import dataclass, field # helps create functions that store data 
from typing import Optional # Optional helps to specify that the param can also be None

import datasets # can load datasets from hugging face
import numpy as np
from datasets import load_dataset # load dataset in -> https://github.com/huggingface/datasets/blob/main/src/datasets/load.py

import transformers
from transformers import (
    AutoConfig, # auto loads a pre-trained model config (e.g. # of layers) from huggingface
    AutoModelForSequenceClassification, # auto loads a pre-trained model from huggingface for sequence classification
    AutoTokenizer, # auto loads a tokenizer from huggingface (tokenizers convert raw text into a format that you can input into a model)
    DataCollatorWithPadding, # collates samples into batches and does padding
    EvalPrediction, # for each input, gives predicted scores for each output class
    EarlyStoppingCallback, # import early stopping
    HfArgumentParser, # a sub-class of `argparse.ArgumentParser`, helps configure arguments to be passed to command-line
    PretrainedConfig, # is the base class for pre-trained models, provides common configs/attributes/methods
    Trainer, # does a lot of the training and evaluation
    TrainingArguments, # pass this in to an instance of Trainer for argument configurations
    default_data_collator, # pass this to DataLoader() to collate input data and do padding
    set_seed, # setting a random seed
)
from transformers.trainer_utils import get_last_checkpoint # returns the path to the last saved checkpoint file during training (useful if training is interrupted)
from transformers.utils import check_min_version # checks if transformers package meets minimum version requirements
# from transformers.utils.versions import require_version # raises an error if version does not meet minimum requirements

import json

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

logger = logging.getLogger(__name__) # creates a logger object with the "name" get_best_model, to identify where these logger messages are coming from

# Get the current working directory
cwd = os.getcwd()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)

@dataclass
class DataTrainingArguments:
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

    def __post_init__(self): # does additional checks after defining params
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need a training/validation/file.")
        else:
            train_extension = self.train_file.split(".")[-1] # gets the document type e.g. "csv" or "json". makes sure that it is
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or json file."
            validation_extension = self.validation_file.split(".")[-1] # e.g. "csv" or "json". makes sure that it is
            assert validation_extension == train_extension, "`validation_file` should have the same extension as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field( # path to pre-trained model
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
    Arguments for training/optimizing data in addition to the default params in TrainingArguments() from transformers.
    """

    seed: int = field(
        default = 123,
        metadata = {"help": "the seed to set"}
    )
    use_mps_device: bool = field(
        default = True,
        metadata = {"help": "whether or not to use mps device"}
    )

def main():

    # takes command-line arguments and parses them into dataclasses -> data_args, model_args, training_args
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, MyTrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # set attributes for early stopping patience
    setattr(training_args, "load_best_model_at_end", True)
    setattr(training_args, "evaluation_strategy", "steps")
    setattr(training_args, "save_strategy", "steps")
    setattr(training_args, "metric_for_best_model", "eval_loss")
    setattr(training_args, "eval_steps", 200)

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

    # Log on each process the small summary:
    logger.warning( # warning message with info about process rank, # of gpus being used, etc.
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}") # log message of all the params needed for training and evaluation

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir: # if there is a prior output directory, and overwrite_output_dir = False, then it tries to find last checkpoint saved in the directory
        last_checkpoint = get_last_checkpoint(training_args.output_dir) # gets the last checkpoint in that output directory
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0: # raises an error if no checkpoint is found and there are files in the output directory
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None: # if checkpoint is found but resume_from_checkpoint is not set, tells the user it will go ahead and resume from latest checkpoint
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # get best hyperparameters
    with open("../results/hyperparameter_optimization/optimized_hyperparameters.json", "r") as f:
        optimized_hyperparameters = json.load(f)

    # load best_hyperparameters into training_args
    for k, v in optimized_hyperparameters.items():
        if k == "best_hyperparameters":
            for k2, v2 in v.items():
                setattr(training_args, k2, v2)

    # get dataset file paths from your own local files. csv/json training/evaluation files are needed
    data_files = {"train": '../data/clean/' + data_args.train_file, "validation": '../data/clean/' + data_args.validation_file, "test": '../data/clean/' + data_args.test_file} # file paths to train and validation data, `data_files` is a dictionary with keys

    for key in data_files.keys(): # logs file paths for "train", "validation" in `data_files`
        logger.info(f"load a local file for {key}: {data_files[key]}")

    # load dataset from csv files
    if data_args.train_file.endswith('.csv'):
        clean_datasets = load_dataset(
            "csv", # file extension as first argument
            data_files = data_files, # train/valid/test paths are passed here
            cache_dir = model_args.cache_dir, # cache_dir specifies path to directory where downloaded datasets will be cached
            use_auth_token = True if model_args.use_auth_token else None, # authenticates user
        )
    # load dataset from json files
    else:
        clean_datasets = load_dataset(
            "json", # file extension as first argument
            data_files=data_files, # train/valid/test file paths are passed here
            cache_dir=model_args.cache_dir, # directory where datasets will be cached
            use_auth_token=True if model_args.use_auth_token else None, # authenticates user
        )
    
    # labels
    label_list = clean_datasets["train"].unique("label") # e.g. array([0,2,1])
    label_list.sort()  # e.g. array([0,1,2]) sort it for determinism (to make the results consistent and reproducible when the order of label_list is the same every time)
    num_labels = len(label_list) # e.g. "3"

    # load pre-trained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path, # if .config_name is not None, then pretrained_model_name_or_path = .config_name, else pretrained_model_name_or_path = .model_name_or_path
        num_labels=num_labels, # # of classes
        finetuning_task = "text-classification", # specify the task that the pre-trained model will be fine-tuned on
        cache_dir=model_args.cache_dir, # where pre-trained model will be stored/cached
        revision=model_args.model_revision, # specific model version to use e.g. `abcdefg1234567`
        use_auth_token=True if model_args.use_auth_token else None, # necessary if pre-trained model config is hosted on private model hub like huggingface hub 
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, # if .tokenizer_name is not None, then pretrained_model_name_or_path = .tokenizer_name, else pretrained_model_name_or_path = .model_name_or_path
        cache_dir=model_args.cache_dir, # where to cache/store the downloaded tokenizer
        use_fast=model_args.use_fast_tokenizer, # whether or not to use fast tokenizer
        revision=model_args.model_revision, # e.g. `abcdefg1234567`
        use_auth_token=True if model_args.use_auth_token else None, # necessary if tokenizer is hosted on a private model hub like huggingface hub
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, # pretrained_model_name_or_path = .model_name_or_path
        from_tf=bool(".ckpt" in model_args.model_name_or_path), # specifies if pre-trained model was original saved in tensorflow format (necessary if it was saved in tf format and pytorch version of model is being loaded)
        config=config, # the config object from AutoConfig.from_pretrained()
        cache_dir=model_args.cache_dir, # specifies directory where pre-trained model will be cached/stored
        revision=model_args.model_revision, # e.g. `abcdefg1234567`
        use_auth_token=True if model_args.use_auth_token else None, # necessary if pre-trained model is hosted on a private model hub like huggingface hub
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes, # whether to ignore size mismatches between pre-trained model and config object here, default = False
    )

    # preprocess the clean datasets (identifying the relevant columns)
    non_label_column_names = [name for name in clean_datasets["train"].column_names if name != "label"] # get the non-label column names
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names: # if your non-label column names are "sentence1" and "sentence2", then assumes that there are your non-label column names and assigns them to sentence1_key and sentence2_key
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2: # if non-label column names are NOT "sentence1" and "sentence2", then it checks if there are >= 2 non-label columns
            sentence1_key, sentence2_key = non_label_column_names[:2] # sets first 2 non-label column names as sentence1_key and sentence2_key (because [:2] is 0,1 elements, excluding 2)
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None # if there are < 2 non-label columns, 1 non-label column name is set to sentence1_key, and sentence2_key = None
    
    # Some models have set the order of the labels to use, so let's make sure we do use it. 
    # Overall, the below code ensures that the labels/ordering used in pre-trained model's config matches labels/ordering in training data
    # "label to id" means you map each possible label to a unique ID
    label_to_id = None 
    if (config.label2id != PretrainedConfig(num_labels=num_labels).label2id): # e.g. default PretrainedConfig().label2id mapping -> {0:'LABEL_0', 1:'LABEL_1', 2:'LABEL_2'}, e.g.model.config.label2id -> {"NEUTRAL": 0, "CONTRADICTION": 1, "ENTAILMENT": 2})
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()} # make the label keys lowercase in model.config: for k, v in model.config.label2id.items() -> e.g. for k, v in [("NEUTRAL", 0), ("CONTRADICTION", 1), ("ENTAILMENT", 2)], `label_name_to_id` = {k.lower(): v} as a dictionary -> {neutral:0, contradiction:1, entailment:2}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)): # e.g. sorted(label_name_to_id.keys()) = ["contradiction", "entailment", "neutral"], e.g. sorted(label_list) = ["contradiction", "entailment", "neutral"]
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)} # e.g. if `label_list` = [entailment, neutral, contradiction], then `label_to_id` = {0:label_name_to_id["entailment"], 1:label_name_to_id["neutral"], 2:label_name_to_id["contradiction"]} -> = {0:2, 1:0, 2:1} ("0 is the entailment in training label_list while 2 is the entailment in model.config labels")
            # e.g. label_to_id = {0:2, 1:0, 2:1}, where training label "0" maps to model.config label "2"
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    label_to_id = {v: i for i, v in enumerate(label_list)} # e.g. label_to_id = {entailment:0, neutral:1, contradiction:2}
    
    # padding strategy
    padding = "max_length" # will pad all samples to `max_seq_length`

    # preprocesses clean data into tokens to input into the model
    def preprocess_function(examples, max_seq_length): 
        """
        takes a batch of examples and tokenizes them 
        with padding (make sure input seq are the same length), 
        max_seq_length (determines the max length of tokenized sequences), 
        and truncation (truncate/shortens sequences that are longer than max_seq_length).
        """
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]) 
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True) # tokenizer() takes input examples sentence1 and sentence2 and -> {'input_ids':[101, 1996, 4248, 2829, 4419, 0, 0], 'attention_mask':[1,1,1,1,1,0,0], 'token_type_ids':[0,0,0,1,1,0,0]}
        # e.g. examples[sentence1_key] = ['The cat in the hat.', 'The fox in the shoe.', 'The dwarf in the mug.'] (this is a list of sentence1's for a given batch)
        # e.g. examples[sentence2_key] = ['The dog in the mut.', 'The woof in the bark.', 'The giant in the cave.'] (this is a list of sentence2's for a given batch)
        # tokenizer from AutoTokenizer.from_pretrained() above

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]] # e.g. for l in examples["label"] = [0,1,2,0,2,1,1]. checks that the label is not unknown (-1 represents unknown or NaN)
        return result # e.g. results for a batch of 2 sentence pair examples {'input_ids':[[101, 1996, 4248, 2829, 4419, 0, 0], [92, 903, 384, 2700, 0, 0, 0]]'attention_mask':[[1,1,1,1,1,0,0],[1,1,1,1,0,0,0]], 'token_type_ids':[[0,0,0,1,1,0,0],[0,0,1,1,0,0,0]], 'label':[1,0]}
    
    # if max_seq_length is greater than the max allowed length for the tokenizer, gives a warning... 
    if training_args.max_seq_length > tokenizer.model_max_length: 
        logger.warning(
        f"The max_seq_length passed ({training_args.max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
    max_seq_length = min(training_args.max_seq_length, tokenizer.model_max_length) # ...and sets the max_seq_length to the max allowed length for tokenizer

    # preprocess the clean datasets into tokens
    with training_args.main_process_first(desc="dataset map pre-processing"): # desc is a description string, .main_process_first ensures that code within it is executed only on the main process during distributed training
        clean_datasets = clean_datasets.map( # maps the pre-process function (tokenizer) to batches of input examples of data
            lambda examples: preprocess_function(examples, max_seq_length),
            batched=True,
            load_from_cache_file=True, # load data from a cache file (speeds up dataset loading times)
            desc="Running tokenizer on dataset", # desc is description string 
        )
    
    # get train dataset
    train_dataset = clean_datasets["train"]

    # get the eval dataset
    eval_dataset = clean_datasets["validation"]

    # get the test dataset
    # test_dataset = clean_datasets["test"]

    # save the pre-processed train/eval/test data
    # train_dataset.save_to_disk('../data/preprocessed/preprocessed_train')
    # eval_dataset.save_to_disk('../data/preprocessed/preprocessed_eval')
    # test_dataset.save_to_disk('../data/preprocessed/preprocessed_test')

    def compute_metrics(p: EvalPrediction): # p or EvalPrediction is a tuple with `predictions`: predicted probs of the model and `label_ids`: true labels
        """
        takes an `EvalPrediction` object (a namedtuple with a predictions and label_ids field)
        and returns a dictionary with a single key value pair, e.g. {"accuracy": 0.85}
        """
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # isinstance() checks if it is a tuple, this line extracts the model predicted probs
        preds = np.argmax(preds, axis=1) # change pred format to be suitable for computing evaluation metrics
        # np.argmax() gets the index of the highest probability class label predicted by model
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()} # gets an array of True/False, .astype(np.float32) converts this to 1 or 0, then take the mean for accuracy

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change to default because we will already do the padding in preprocess_function
    data_collator = default_data_collator # change to default data collator because we set `padding = "max_length" above``
    if training_args.fp16: # if .fp16 = True (using float16 precision can speed up training by reducing memory requirements and allowing for larger batch sizes)
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) # pad to multiple of 8 because some hardware perform better when input sequences are padded to 8
    
    # Log a few random samples from the training set: (to quickly inspect contents of training data during training if needed)
    for index in random.sample(range(len(train_dataset)), 3): # randomly samples 3 row indices
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.") # logs 3 training rows/examples

    # create best trainer
    best_trainer = Trainer(
        model=model, # pre-trained model to be fine-tuned
        args=training_args, # contains arguments from TrainingArguments, including task_name
        train_dataset=train_dataset, # pre-processed training dataset (e.g. dict of lists of input_ids, attention_mask, token_type_ids, labels)
        eval_dataset=eval_dataset, # pre-processed validation dataset (e.g. dict of lists of input_ids, attention_mask, token_type_ids, labels)
        compute_metrics=compute_metrics, # above function to evaluate metrics
        tokenizer=tokenizer, # specifies the tokenizer to be used by the model to make predictions on new data (not to pre-process input data because you already did that in preprocess_function)
        data_collator=data_collator, # specify the kind of data collator you want, default is `DataCollatorWithPadding` (see data_collator code above) (data collator is used to convert examples to batches)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # training stops if there is no improvement for 3 consecutive evaluations (prevents overfitting)
    )

    # Training
    logger.info("*** Training Full Dataset ***")
    logger.info(f"Starting training with optimized hyperparameters: learning_rate = {training_args.learning_rate: .5f}, per_device_train_batch_size = {training_args.per_device_train_batch_size}, num_train_epochs = {training_args.num_train_epochs}, max_seq_length = {max_seq_length}, weight_decay = {training_args.weight_decay: .5f}.")

    checkpoint = None # init checkpoint
    if training_args.resume_from_checkpoint is not None: # if resume_from_checkpoint
        checkpoint = training_args.resume_from_checkpoint # sets the checkpoint to the path of a saved checkpoint
    elif last_checkpoint is not None: 
        checkpoint = last_checkpoint # sets checkpoint to path to last checkpoint
    train_result = best_trainer.train(resume_from_checkpoint=checkpoint) # feeds checkpoint if avail, if not then starts from scratch
    # during training, Trainer() updates the model's weights, and evaluates/validates model performance on training data after each epoch
    # e.g. train_result = TrainingOutput(global_step = 1000, training_loss = 0.25, learning_rate = 0.001, epoch = 5, metrics = {"accuracy":0.85, "f1":0.83})
    # global step: total # of training steps (one update of model weights) that were run, training_loss: avg training loss over all batches, learning_rate: learning rate used during training, epoch: # of epochs (epoch = 1 iteration on entire dataset) that were run, metrics: dict of training metrics
    metrics = train_result.metrics # get metrics
    metrics["train_samples"] = len(train_dataset) # put # of train samples key in metrics dictionary

    # Trainer() methods 
    best_trainer.save_model()  # Saves the tokenizer too for easy upload, saves model weights and config to disk in file/format that can be loaded later for inference or training
    best_trainer.log_metrics("train", metrics) # logs the training metrics, e.g. {"accuracy":0.85, "f1":0.83, "train_samples":500}
    best_trainer.save_metrics("train", metrics) # saves the training metrics to a file on the disk 
    best_trainer.save_state() # saves the entire state (e.g. current iteration, optimizer state, scheduler state) of the Trainer() object ot the disk, allows training to be resumed in case of interruptions

def _mp_fn(index): # used in conjunction with the below code when running on TPUs, takes in `index` argument which is the process index used by `xla_spawn()`
    # For xla_spawn (TPUs)
    main()
# e.g. for using TPUs
# import torch_xla.distributed.xla_multiprocessing as xmp
# if __name__ == '__main__':
#   xmp.spawn(_mp_fn, args=(), nprocs=8)
    
if __name__ == "__main__": # if the script is run from the command line, sets __name__ to "__main__" and main() will be executed. if the script is imported into another script, `__name__` will be set to `get_best_model` and main() will not be executed
    main()