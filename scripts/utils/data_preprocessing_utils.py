import logging
from typing import List, Tuple

import datasets
from transformers import PretrainedConfig, PreTrainedTokenizer


def get_train_subsets(clean_datasets, data_args, training_args):
    """
    Extracts a subset of the training dataset based on the training_args.train_subset_perc parameter.

    Args:
        clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
        data_args (DataArguments): The data arguments containing dataset-related configurations.
        training_args (TrainingArguments): The training arguments containing training-related configurations.

    Returns:
        datasets.Dataset: The subset of the training dataset.
    """
    train_dataset = clean_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    short_train = train_dataset.select(range(int(training_args.train_subset_perc * len(train_dataset))))
    return short_train

def get_eval_subsets(clean_datasets, data_args, training_args):
    """
    Extracts a subset of the evaluation dataset based on the training_args.eval_subset_perc parameter.

    Args:
        clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
        data_args (DataArguments): The data arguments containing dataset-related configurations.
        training_args (TrainingArguments): The training arguments containing training-related configurations.

    Returns:
        datasets.Dataset: The subset of the evaluation dataset.
    """
    eval_dataset = clean_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    short_eval = eval_dataset.select(range(int(training_args.eval_subset_perc * len(eval_dataset))))
    return short_eval

def get_subsets(logger, clean_datasets, data_args, training_args):
    """
    Gets the subsets of train and eval datasets based on the specified training and evaluation percentage parameters.

    Args:
        logger (logging.Logger): The logger to display information during the process.
        clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
        data_args (DataArguments): The data arguments containing dataset-related configurations.
        training_args (TrainingArguments): The training arguments containing training-related configurations.

    Returns:
        Tuple[datasets.Dataset, datasets.Dataset]: The subsets of the training and evaluation datasets.
    """
    logger.info(f"Getting subsets...")

    short_train = get_train_subsets(clean_datasets, data_args, training_args)
    short_eval = get_eval_subsets(clean_datasets, data_args, training_args)
    return short_train, short_eval

def get_label_information(logger, clean_datasets: datasets.DatasetDict) -> Tuple[List[str], int]:
    """
    Gets label information, including the unique labels and the number of labels.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        clean_datasets (datasets.DatasetDict): A dictionary containing the clean datasets for train, validation, and test sets.

    Returns:
        tuple: A tuple containing a sorted list of unique labels and the number of labels.
    """
    logger.info(f"Getting label information...")

    label_list = clean_datasets["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)
    return label_list, num_labels

def get_max_seq_length(max_seq_length, tokenizer, logger):
    """
    Args:
        max_seq_length (int): the max allowed length for the tokenizer
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): the tokenizer
        logger (logging.Logger): the logger

    Returns:
        max_seq_length (int): the max allowed length for the tokenizer
    """
    logger.info(f"Getting max_seq_length...")

    if max_seq_length > tokenizer.model_max_length: # if max_seq_length is greater than the max allowed length for the tokenizer, gives a warning... 
        logger.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
        max_seq_length = min(max_seq_length, tokenizer.model_max_length) # ...and sets the max_seq_length to the max allowed length for tokenizer
    return max_seq_length 
 
def get_non_label_column_names(logger, clean_datasets):
    """
    Gets the column names in the dataset that are not 'label'.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        clean_datasets (datasets.DatasetDict): A dictionary containing the clean datasets for train, validation, and test sets.

    Returns:
        list: A list of column names that are not 'label'.
    """
    logger.info(f"Getting non_label_column_names...")

    non_label_column_names = [name for name in clean_datasets["train"].column_names if name != "label"]
    return non_label_column_names

def get_sentence_keys(logger, non_label_column_names):
    """
    Gets the sentence1_key and sentence2_key based on the non_label_column_names.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        non_label_column_names (list): A list of column names that are not 'label'.

    Returns:
        tuple: A tuple containing the sentence1_key and sentence2_key.
    """
    logger.info(f"Getting sentence1_key, sentence2_key...")

    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None
    return sentence1_key, sentence2_key

def get_label_to_id(logger, config, num_labels, label_list):
    """
    Gets the mapping of labels to IDs.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        config (PretrainedConfig): The pre-trained model configuration.
        num_labels (int): The number of unique labels in the dataset.
        label_list (list): A list of unique labels.

    Returns:
        dict: A dictionary containing the mapping of labels to IDs.
    """
    logger.info(f"Getting label_to_id...")

    # this function is necessary because the label IDs are used internally by the model during training, but the label names are used externally to interpret the model's predictions.
    # it makes sure that the internal representations match the external representations
    label_to_id = None
    if (config.label2id != PretrainedConfig(num_labels=num_labels).label2id):
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    label_to_id = {v: i for i, v in enumerate(label_list)}
    return label_to_id

def preprocess_clean_datasets(logger, clean_datasets, config, num_labels, label_list):
    """
    Preprocesses the clean datasets and gets the sentence keys and label_to_id mapping.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        clean_datasets (datasets.DatasetDict): A dictionary containing the clean datasets for train, validation, and test sets.
        config (PretrainedConfig): The pre-trained model configuration.
        num_labels (int): The number of unique labels in the dataset.
        label_list (list): A list of unique labels.

    Returns:
        tuple: A tuple containing the sentence1_key, sentence2_key, and label_to_id mapping.
    """
    logger.info(f"Preprocessing clean datasets...")

    non_label_column_names = get_non_label_column_names(logger, clean_datasets)
    sentence1_key, sentence2_key = get_sentence_keys(logger, non_label_column_names)
    label_to_id = get_label_to_id(logger, config, num_labels, label_list)
    return sentence1_key, sentence2_key, label_to_id

# preprocesses clean data into tokens to input into the model
def preprocess_tokenize_function(example, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer): 
    """
    Preprocesses and tokenizes a single example.

    Args:
        example (dict): A single example from the dataset.
        max_seq_length (int): The maximum sequence length allowed.
        sentence1_key (str): The key for the first sentence in the example.
        sentence2_key (str): The key for the second sentence in the example, if applicable.
        label_to_id (dict): A dictionary containing the mapping of labels to IDs.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.

    Returns:
        dict: A dictionary containing the tokenized input IDs, attention masks, token type IDs, and label IDs.
    """
    # Tokenize the texts
    args = (
        (example[sentence1_key],) if sentence2_key is None else (example[sentence1_key], example[sentence2_key]) 
    )
    result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True) # tokenizer() takes input example sentence1 and sentence2 and -> {'input_ids':[101, 1996, 4248, 2829, 4419, 0, 0], 'attention_mask':[1,1,1,1,1,0,0], 'token_type_ids':[0,0,0,1,1,0,0]}
    # e.g. example[sentence1_key] = ['The cat in the hat.', 'The fox in the shoe.', 'The dwarf in the mug.'] (this is a list of sentence1's for a given batch)
    # e.g. example[sentence2_key] = ['The dog in the mut.', 'The woof in the bark.', 'The giant in the cave.'] (this is a list of sentence2's for a given batch)

    # Map labels to IDs 
    if label_to_id is not None and "label" in example:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in example["label"]] # e.g. for l in example["label"] = [0,1,2,0,2,1,1]. checks that the label is not unknown (-1 represents unknown or NaN)
    return result # e.g. results for a batch of 2 sentence pair examples {'input_ids':[[101, 1996, 4248, 2829, 4419, 0, 0], [92, 903, 384, 2700, 0, 0, 0]]'attention_mask':[[1,1,1,1,1,0,0],[1,1,1,1,0,0,0]], 'token_type_ids':[[0,0,0,1,1,0,0],[0,0,1,1,0,0,0]], 'label':[1,0]}

def preprocess_datasets(logger, clean_datasets, training_args, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer): # don't need to feed in example because it's a built in variable name used in the `map` method of hugging face datasets library used to represent an individual example of `clean_datasets`, which is a datasets library object
    """
    Applies the preprocess_tokenize_function to all examples in the clean datasets.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        clean_datasets (datasets.DatasetDict): A dictionary containing the clean datasets for train, validation, and test sets.
        training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.
        max_seq_length (int): The maximum sequence length allowed.
        sentence1_key (str): The key for the first sentence in the examples.
        sentence2_key (str): The key for the second sentence in the examples, if applicable.
        label_to_id (dict): A dictionary containing the mapping of labels to IDs.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.

    Returns:
        datasets.DatasetDict: The preprocessed and tokenized datasets for train, validation, and test sets.
    """
    logger.info(f"Applying preprocess_tokenize_function to examples in clean datasets...")

    with training_args.main_process_first(desc="dataset map pre-processing"):
        clean_datasets = clean_datasets.map( # map applies a function to all the examples in a dataset
            lambda example: preprocess_tokenize_function(example, max_seq_length, sentence1_key, sentence2_key, label_to_id, tokenizer), # takes in an example, and applies the preprocess function to it
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    return clean_datasets

