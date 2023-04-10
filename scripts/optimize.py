import logging # logs msgs/errors 

from transformers import (
     set_seed # setting a random seed
)

from transformers.utils import check_min_version # checks if transformers package meets minimum version requirements

from utils.args_utils import *
from utils.logging_utils import *
from utils.checkpoint_utils import *
from utils.data_load_and_clean_utils import *
from utils.data_preprocessing_utils import *
from utils.hyperparam_utils import *
from utils.model_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

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
    save_optimized_hyperparams(logger, study)

if __name__ == "__main__": # if the script is run from the command line, sets __name__ to "__main__" and main() will be executed. if the script is imported into another script, `__name__` will be set to `run_glue` and main() will not be executed
    main()