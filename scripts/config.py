import os
from utils.args_utils import *

# Get args
data_args, model_args, training_args = parse_arguments()

# Get the parent directory
PAR_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# Get the root directory
ROOT_DIR = os.path.join(PAR_DIR, 'MedNLI-Text-Classification/')

# Define data directories
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
CLEAN_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'clean')

# Define raw data file paths
RAW_TRAIN_FILE = os.path.join(RAW_DATA_DIR, data_args.train_file)
RAW_VALIDATION_FILE = os.path.join(RAW_DATA_DIR, data_args.validation_file)
RAW_TEST_FILE = os.path.join(RAW_DATA_DIR, data_args.test_file)

# Define clean data file paths
CLEAN_TRAIN_FILE = os.path.join(CLEAN_DATA_DIR, data_args.train_file)
CLEAN_VALIDATION_FILE = os.path.join(CLEAN_DATA_DIR, data_args.validation_file)
CLEAN_TEST_FILE = os.path.join(CLEAN_DATA_DIR, data_args.test_file)

# Define optimized hyperparameters file path
OPTIMIZED_HYPERPARAMS_DIR = os.path.join(ROOT_DIR, 'results', 'hyperparameter_optimization')

# Define fine-tuned model file path
FINE_TUNED_MODEL_DIR = os.path.join(ROOT_DIR, 'results', 'training')

# Define the evaluation results file path
EVAL_RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'evaluation')