import os
import logging # logs msgs/errors 

from transformers import set_seed
from transformers.utils import check_min_version # checks if transformers package meets minimum version requirements

from utils.args_utils import parse_arguments
from utils.logging_utils import *
from utils.data_load_and_clean_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

def main():

    # Parse arguments
    data_args, model_args, training_args = parse_arguments()

    # Setup logging
    logger = logging.getLogger(__name__) # creates a logger object with the "name" get_best_model, to identify where these logger messages are coming from
    set_logger_and_verbosity(logger)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load and clean data
    train_list, eval_list, test_list = load_and_clean_data(logger, data_args)

    # export data as json files
    export_clean_data(logger, train_list, eval_list, test_list, os) 

if __name__ == "__main__": # if the script is run from the command line, sets __name__ to "__main__" and main() will be executed. if the script is imported into another script, `__name__` will be set to `run_glue` and main() will not be executed
    main()
