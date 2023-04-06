def create_study(logger):
    """
    Creates an Optuna study for hyperparameter optimization.

    Args:
        logger (logging.Logger): A logger object for logging messages.

    Returns:
        optuna.study.Study: An Optuna study object for hyperparameter optimization.
    """
    logger.info(f"Creating study...")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), study_name="Optimize Hyperparameters")
    study.enqueue_trial({"learning_rate": 3e-5, "per_device_train_batch_size": 8, "num_train_epochs": 3, "max_seq_length": 128, "weight_decay": 1e-2})
    return study

def get_search_space(param_name):
    """
    Defines the search space for hyperparameter optimization.

    Args:
        param_name (str): The name of the hyperparameter for which to get the search space.

    Returns:
        tuple: A tuple containing the search space for the specified hyperparameter.
    """
    search_space = {
        "learning_rate": (2e-5, 5e-5),
        "per_device_train_batch_size": (8, 16),
        "num_train_epochs": (2, 4),
        "max_seq_length": (128, 384),
        "weight_decay": (1e-4, 1e-2)
    }
    return search_space[param_name]

def get_hyperparameters(trial, get_search_space, logger):
    """
    Gets the hyperparameters to try from the search space and logs the current trial number and hyperparameter values.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object for the current trial.
        get_search_space (function): A function to get the search space for a specific hyperparameter.
        logger (logging.Logger): A logger object for logging messages.

    Returns:
        tuple: A tuple containing the suggested hyperparameter values.
    """
    learning_rate = trial.suggest_loguniform("learning_rate", *get_search_space("learning_rate")) # e.g. trial.suggest_loguniform() accesses the trainer object and modifies its learning rate
    per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", *get_search_space("per_device_train_batch_size"))
    num_train_epochs = trial.suggest_int("num_train_epochs", *get_search_space("num_train_epochs"))
    max_seq_length = trial.suggest_int("max_seq_length", *get_search_space("max_seq_length"), step=64)
    weight_decay = trial.suggest_loguniform("weight_decay", *get_search_space("weight_decay"))
    
    logger.info(f"Starting trial {trial.number} with learning_rate = {learning_rate:.5f}, per_device_train_batch_size = {per_device_train_batch_size}, num_train_epochs = {num_train_epochs}, max_seq_length = {max_seq_length}, weight_decay = {weight_decay:.5f}.")
    
    return learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay

def update_training_args_with_hyperparams_and_early_stopping(logger, training_args, learning_rate, per_device_train_batch_size, num_train_epochs, max_seq_length, weight_decay):
    """
    Sets the training arguments to the suggested hyperparameter values and configures early stopping.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.
        learning_rate (float): The suggested learning rate value.
        per_device_train_batch_size (int): The suggested per-device train batch size value.
        num_train_epochs (int): The suggested number of training epochs.
        max_seq_length (int): The suggested maximum sequence length value.
        weight_decay (float): The suggested weight decay value.

    Returns:
        MyTrainingArguments: The updated instance of MyTrainingArguments with the suggested hyperparameter values.
    """
    logger.info(f"Setting training_args attributes to suggested values.")
    
    # set attributes for early stopping patience
    setattr(training_args, "load_best_model_at_end", True)
    setattr(training_args, "evaluation_strategy", "steps")
    setattr(training_args, "save_strategy", "steps")
    setattr(training_args, "metric_for_best_model", "eval_loss")
    setattr(training_args, "eval_steps", 200)

    # feed the suggested values into training_args
    setattr(training_args, "learning_rate", learning_rate)
    setattr(training_args, "per_device_train_batch_size", per_device_train_batch_size)
    setattr(training_args, "num_train_epochs", num_train_epochs)
    setattr(training_args, "max_seq_length", max_seq_length)
    setattr(training_args, "weight_decay", weight_decay)
    return training_args

# define the hyperparameter optimization function
def objective(trial, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets):
    """
    The objective function for the hyperparameter optimization process.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object that contains trial information and hyperparameters.
        data_args (DataArguments): The data arguments containing dataset-related configurations.
        model_args (ModelArguments): The model arguments containing model-related configurations.
        training_args (TrainingArguments): The training arguments containing training-related configurations.
        logger (logging.Logger): The logger to display information during the process.
        last_checkpoint (str): The path to the last checkpoint from which to resume training.
        clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.

    Returns:
        float: The accuracy of the model on the evaluation dataset.
    """
    return train_and_evaluate(
            trial=trial,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            logger=logger,
            last_checkpoint=last_checkpoint,
            clean_datasets=clean_datasets
        )

def optimize_hyperparameters(study, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets):
    """
    Optimizes the hyperparameters using the objective function and Optuna.

    Args:
        study (optuna.study.Study): An Optuna study object to manage the optimization process.
        data_args (DataArguments): The data arguments containing dataset-related configurations.
        model_args (ModelArguments): The model arguments containing model-related configurations.
        training_args (TrainingArguments): The training arguments containing training-related configurations.
        logger (logging.Logger): The logger to display information during the process.
        last_checkpoint (str): The path to the last checkpoint from which to resume training.
        clean_datasets (datasets.DatasetDict): The preprocessed and tokenized datasets for train, validation, and test sets.
    """
    study.optimize(lambda trial: objective(trial, data_args, model_args, training_args, logger, last_checkpoint, clean_datasets),
                   n_trials=training_args.n_trials)
    return study

def save_optimized_hyperparams(logger, study, output_path):
    """
    Saves the optimized hyperparameters as a JSON file.

    Args:
        logger (logging.Logger): The logger to display information during the process.
        study (optuna.study.Study): An Optuna study object to manage the optimization process.
        output_path (str): The path to the output file where the optimized hyperparameters will be saved.
    """
    logger.info(f"Saving optimized hyperparameters...")

    optimized_hyperparams = {
        'best_hyperparameters': study.best_params,
        'best_accuracy': study.best_value
    }
    with open(output_path, 'w') as f:
        json.dump(optimized_hyperparams, f)

def load_optimized_hyperparameters(filepath, logger):
    """
    Load optimized hyperparameters from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing optimized hyperparameters.

    Returns:
        dict: A dictionary containing the optimized hyperparameters.
    """
    logger.info(f"Loading optimized hyperparameters from {filepath}...")

    with open(filepath, "r") as f:
        optimized_hyperparameters = json.load(f)
    return optimized_hyperparameters

def set_optimized_hyperparameters(logger, training_args, optimized_hyperparameters):
    """
    Set the optimized hyperparameters to the training arguments.

    Args:
        training_args (TrainingArguments): The training arguments containing training-related configurations.
        optimized_hyperparameters (dict): A dictionary containing the optimized hyperparameters.

    Returns:
        TrainingArguments: The updated training arguments with optimized hyperparameters.
    """
    logger.info(f"Setting training_args attributes to optimized hyperparameter values...")

    for k, v in optimized_hyperparameters.items():
        if k == "best_hyperparameters":
            for k2, v2 in v.items():
                setattr(training_args, k2, v2)
    return training_args

def update_training_args_with_early_stopping(logger, training_args, clean_datasets):
    """
    Sets the training arguments to the suggested hyperparameter values and configures early stopping.

    Args:
        logger (logging.Logger): A logger object for logging messages.
        training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.
    Returns:
        MyTrainingArguments: The updated instance of MyTrainingArguments with the suggested hyperparameter values.
    """
    logger.info(f"Setting training_args attributes to suggested values.")
    
    # set attributes for early stopping patience
    setattr(training_args, "load_best_model_at_end", True)
    setattr(training_args, "evaluation_strategy", "steps")
    setattr(training_args, "save_strategy", "steps")
    setattr(training_args, "metric_for_best_model", "eval_loss")
    train_length = len(clean_datasets["train"])
    setattr(training_args, "eval_steps", int(train_length/training_args.per_device_train_batch_size*0.55)) # total steps per epoch * 0.55 will let you evaluate the model 1 or 2 times per epoch

    return training_args

