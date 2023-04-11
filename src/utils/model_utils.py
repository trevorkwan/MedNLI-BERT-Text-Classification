import logging
import os

import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    default_data_collator,
)

from config import FINE_TUNED_MODEL_DIR

# need to set logger in helper function scripts
logger = logging.getLogger(__name__)

def load_config_tokenizer_model(logger, model_args, num_labels):
    """
    Loads the pre-trained configuration, tokenizer, and model.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        model_args (ModelArguments): An instance of ModelArguments containing the model arguments.
        num_labels (int): The number of unique labels in the dataset.

    Returns:
        tuple: A tuple containing the configuration, tokenizer, and model objects.
    """
    logger.info(f"Loading pre-trained config, tokenizer, and model...")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    return config, tokenizer, model

def compute_metrics(p: EvalPrediction): # p or EvalPrediction is a tuple with `predictions`: predicted probs of the model and `label_ids`: true labels
    """
    Computes the accuracy of the predictions against the true labels.

    Args:
        p (EvalPrediction): A named tuple containing two fields: predictions and label_ids.

    Returns:
        Dict[str, float]: A dictionary with a single key-value pair representing the accuracy.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # isinstance() checks if it is a tuple, this line extracts the model predicted probs
    preds = np.argmax(preds, axis=1) # change pred format to be suitable for computing evaluation metrics
    # np.argmax() gets the index of the highest probability class label predicted by model
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()} # gets an array of True/False, .astype(np.float32) converts this to 1 or 0, then take the mean for accuracy

def create_data_collator(logger, training_args, tokenizer):
    """
    Creates a data collator based on the training arguments and tokenizer.

    Args:
        logger (logging.Logger): The logger to display information during the process.
        training_args (TrainingArguments): The training arguments containing training-related configurations.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used in the data collator.

    Returns:
        DataCollator: The data collator to be used during training and evaluation.
    """
    logger.info(f"Creating data_collator...")

    data_collator = default_data_collator
    if training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    return data_collator

def create_best_trainer(logger, model, training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator):
    """
    Creates a Trainer instance with the best hyperparameters and early stopping callback.

    Args:
        model (PreTrainedModel): The pre-trained model to be fine-tuned.
        training_args (TrainingArguments): An instance of TrainingArguments containing the training arguments.
        train_dataset (datasets.Dataset): The pre-processed training dataset.
        eval_dataset (datasets.Dataset): The pre-processed validation dataset.
        compute_metrics (callable): A function to evaluate the model's metrics.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used by the model.
        data_collator (DataCollator): The data collator to convert examples to batches.

    Returns:
        Trainer: The created Trainer instance.
    """
    logger.info(f"Creating Best Trainer")

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

def train_model(logger, best_trainer, train_dataset, last_checkpoint=None):
    """
    Trains the model using the best Trainer instance.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        best_trainer (Trainer): The best Trainer instance for model training.
        train_dataset (datasets.Dataset): The pre-processed training dataset.
        last_checkpoint (str, optional): The path to the last checkpoint from which to resume training. Defaults to None.

    Returns:
        dict: The training metrics.
    """
    logger.info("*** Training Full Dataset ***")
    logger.info(f"Starting training with optimized hyperparameters: learning_rate = {best_trainer.args.learning_rate: .5f}, per_device_train_batch_size = {best_trainer.args.per_device_train_batch_size}, num_train_epochs = {best_trainer.args.num_train_epochs}, max_seq_length = {best_trainer.args.max_seq_length}, weight_decay = {best_trainer.args.weight_decay: .5f}.")

    checkpoint = last_checkpoint if last_checkpoint is not None else best_trainer.args.resume_from_checkpoint
    train_result = best_trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    return metrics

def save_trainer_state(logger, best_trainer, metrics):
    """
    Saves the trainer state, metrics, and model to disk.

    Args:
        best_trainer (Trainer): The best Trainer instance for model training.
        metrics (dict): The training metrics.
    """
    logger.info(f"Saving trainer state, metrics, and model to disk...")
                
    best_trainer.save_model()
    best_trainer.log_metrics("train", metrics)
    best_trainer.save_metrics("train", metrics)
    best_trainer.save_state()

def get_fine_tuned_model_path(model_args):
    """
    Set the path for the model.

    Args:
        model_args: An instance of a model arguments class.
        path: A string representing the path to the model.

    Returns:
        None
    """
    path = FINE_TUNED_MODEL_DIR
    setattr(model_args, "model_name_or_path", path)

def set_evaluation_metric(training_args, metric_name="eval_accuracy"):
    """
    Sets the metric_for_best_model attribute in the TrainingArguments instance.

    Args:
        training_args (TrainingArguments): An instance of TrainingArguments.
        metric_name (str): The name of the metric to use for early stopping, default is "eval_accuracy".
    """
    training_args.metric_for_best_model = metric_name

def evaluate_test_dataset(logger, trainer, test_dataset):
    """
    Evaluates the model on the test dataset.

    Args:
        trainer (Trainer): A Trainer instance used for model training.
        test_dataset (Dataset): The test dataset to evaluate the model on.

    Returns:
        dict: A dictionary containing the evaluation metrics and the number of test samples.
    """
    logger.info("*** Testing Full Dataset ***")

    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    metrics["test_samples"] = len(test_dataset)
    return metrics

def log_and_save_test_metrics(logger, trainer, metrics):
    """
    Logs and saves the test metrics.

    Args:
        trainer (Trainer): A Trainer instance used for model training.
        metrics (dict): A dictionary containing the evaluation metrics and the number of test samples.
    """
    logger.info(f"Saving evaluation metrics on test dataset...")

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics, combined = False)

def predict_on_test_dataset(logger, trainer, predict_dataset, label_list, output_dir):
    """
    Makes predictions on the full dataset and saves the results to a file.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        trainer (Trainer): A Trainer instance used for model training.
        predict_dataset (Dataset): The dataset to make predictions on.
        label_list (list): A list of label names to map prediction indices to label names.
        output_dir (str): The directory where the prediction results file should be saved.

    Returns:
        None
    """
    logger.info("*** Predicting on Test Dataset ***")

    # Removing the `label` columns because it contains -1 and Trainer won't like that.
    predict_dataset = predict_dataset.remove_columns("label")
    predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    predictions = np.argmax(predictions, axis=1)

    output_predict_file = os.path.join(output_dir, f"predict_results.txt")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Writing predict results *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                item = label_list[item]
                writer.write(f"{index}\t{item}\n")