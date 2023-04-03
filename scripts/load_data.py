import logging # logs msgs/errors 
import os
import sys
import datasets
from datasets import load_dataset

# Get the current working directory
cwd = os.getcwd()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)

logger = logging.getLogger(__name__) # creates a logger object with the "name" get_best_model, to identify where these logger messages are coming from

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # log format
        datefmt="%m/%d/%Y %H:%M:%S", # datetime format
        handlers=[logging.StreamHandler(sys.stdout)], # where to send the log messages e.g. a file, the console
    )
    
    logger.setLevel(logging.INFO) # assigns it to the logger object
    datasets.utils.logging.set_verbosity(logging.INFO) # ensures that log messages emitted by datasets library are `log_level` or higher

    # load imdb
    logger.info(f"Starting to load imdb data.")

    # Load the IMDB dataset
    imdb_dataset = load_dataset('imdb')

    # Print the number of examples in the dataset
    print(f"Number of training examples: {len(imdb_dataset['train'])}")
    print(f"Number of test examples: {len(imdb_dataset['test'])}")

    # get train and test data
    train_dataset = imdb_dataset['train']
    test_dataset = imdb_dataset['test']

    # convert to json and export into raw folder
    train_dataset.to_json("../data/raw/train_imdb.json")
    test_dataset.to_json("../data/raw/test_imdb.json")

    # load imdb
    logger.info(f"Finished loading imdb data and exported as json files to data/raw/ folder.")


if __name__ == "__main__": # if the script is run from the command line, sets __name__ to "__main__" and main() will be executed. if the script is imported into another script, `__name__` will be set to `run_glue` and main() will not be executed
    main()

