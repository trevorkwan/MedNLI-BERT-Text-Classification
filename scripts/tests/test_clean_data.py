import json
import logging
import sys
import os
from unittest.mock import MagicMock

import pytest

# change system path to scripts folder
sys.path.append('../')

# import functions to test from scripts.clean_data.py
from clean_data import (
    set_logger_and_verbosity,
    load_raw_data,
    get_clean_data,
    load_and_clean_data,
    export_json,
    export_clean_data,
)

# create mock classes for DataLoadingArguments and MyTrainingArguments
class DataLoadingArgumentsMock:
    train_file = "train.jsonl"
    validation_file = "validation.jsonl"
    test_file = "test.jsonl"
    max_train_samples = None
    max_eval_samples = None
    max_test_samples = None

class MyTrainingArgumentsMock:
    # mocks the get_process_log_level attribute from MyTrainingArguments training_args.get_process_log_level()
    def get_process_log_level(self):
        return logging.INFO
    # define seed as a mock attribute
    seed = 42

# mock logger object
custom_logger = logging.getLogger("my_custom_logger")

def test_set_logger_and_verbosity():
    """
    Test the `set_logger_and_verbosity` function to ensure the logger's level is set correctly.
    """
    training_args = MyTrainingArgumentsMock()

    # pass the custom logger to the function
    set_logger_and_verbosity(custom_logger)

    assert custom_logger.level == logging.INFO

def test_load_raw_data(tmpdir):
    """
    `load_raw_data` takes in file paths to temporary created data, and we see if it can return the same temporary created data.

    Args:
        tmpdir (py.path.local): A temporary directory object provided by the pytest `tmpdir` fixture.
            Used for creating and storing temporary files during testing.

    """
    data_args = DataLoadingArgumentsMock()

    # Create temporary train, validation, and test files
    train_data = [{"sentence1": "This is a test.", "sentence2": "Is this a test?", "gold_label": "entailment"}]

    # Create temporary dir within the tmpdir
    data_dir = tmpdir.mkdir("data").mkdir("raw")

    # Create temporary train file in the dir and write the content in it
    train_file = data_dir.join("mli_train_v1.jsonl")
    train_file.write(json.dumps(train_data[0]) + "\n")  # Write only the first dictionary in train_data (so that load_raw_data can return train_med as a list of dictionary)

    # Update data_args to use the temporary file paths
    data_args.train_file = os.path.relpath(str(train_file), '../data/raw/') # gets the relative path to "mli_train_v1.jsonl" from the ../data/raw/ directory

    # Create temporary validation and test files (empty for this test)
    validation_file = data_dir.join("mli_validation_v1.jsonl")
    validation_file.write("")
    data_args.validation_file = os.path.relpath(str(validation_file), '../data/raw/')

    test_file = data_dir.join("mli_test_v1.jsonl")
    test_file.write("")
    data_args.test_file = os.path.relpath(str(test_file), '../data/raw/')

    # Test the load_raw_data function
    train_med, eval_med, test_med = load_raw_data(data_args)

    # Check if the loaded data is equal to the original data
    assert train_med == train_data
    assert eval_med == []
    assert test_med == []

def test_get_clean_data():
    """
    Test the `get_clean_data` function by checking if the function correctly extracts and cleans
    the input data, returning a list of dictionaries with the expected structure.
    """
    data = [
        {"sentence1": "This is a test. ", "sentence2": "Is this a test? ", "gold_label": "entailment"}
    ]

    expected_data_list = [
        {"sentence1": "This is a test.", "sentence2": "Is this a test?", "label": "entailment"}
    ]

    data_list = get_clean_data(data)

    assert data_list == expected_data_list

def test_load_and_clean_data(monkeypatch):
    """
    Test the `load_and_clean_data` function by mocking the raw data loading and cleaning functions
    to ensure that the logger info calls are made and the function returns the expected data lists.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for safely mocking attributes and
            functions.
    """
    data_args = DataLoadingArgumentsMock()

    # Mock raw data loading and cleaning
    monkeypatch.setattr("clean_data.load_raw_data", lambda _: ([], [], [])) # replaces `load_raw_data` with a lambda function that takes any input argument returns 3 empty lists (to simulate return raw data)
    monkeypatch.setattr("clean_data.get_clean_data", lambda data, _: []) # lambda function that takes in data and _ ("_" is a placeholder for the second argument) and returns an empty list (to simulate return clean data)

    logger = MagicMock()
    monkeypatch.setattr("clean_data.logger", logger)

    train_list, eval_list, test_list = load_and_clean_data(data_args)

    assert logger.info.call_count == 4 # assert that the logger.info function was called 4 times
    assert train_list == [] # assert that the returned train_list is an empty list
    assert eval_list == []
    assert test_list == []

def test_export_json(tmpdir):
    """
    Test the `export_json` function by checking if the input data is correctly exported as a JSON
    file and can be loaded back with the same content.

    Args:
        tmpdir (py.path.local): A temporary directory object provided by the pytest `tmpdir` fixture.
            Used for creating and storing temporary files during testing.
    """
    data = {"key": "value"}

    file_path = tmpdir.join("output.json")

    export_json(data, str(file_path)) # tests if export_json can take in a dictionary and a file path and export the dictionary as a JSON file

    with open(file_path, "r") as infile: # tests if the JSON file can be loaded back with the same content
        loaded_data = json.load(infile)

    assert loaded_data == data # assert that the loaded data is the same as the original data

def test_export_clean_data(monkeypatch, tmpdir):
    """
    Test the `export_clean_data` function by mocking the JSON exporting function and the logger
    object to ensure the expected number of calls are made and the logger.info method is called.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for safely mocking attributes and
            functions.
        tmpdir (py.path.local): A temporary directory object provided by the pytest `tmpdir` fixture.
            Used for creating and storing temporary files during testing.
    """
    train_list = [{"sentence1": "This is a test.", "sentence2": "Is this a test?", "label": "entailment"}]
    eval_list = [{"sentence1": "I have a pen.", "sentence2": "I have a pencil.", "label": "contradiction"}]
    test_list = [{"sentence1": "The sky is blue.", "sentence2": "The sky is not blue.", "label": "neutral"}]

    # Mock JSON exporting function
    export_json_mock = MagicMock()
    monkeypatch.setattr("clean_data.export_json", export_json_mock)

    # Mock logger object
    logger = MagicMock()
    monkeypatch.setattr("clean_data.logger", logger)

    export_clean_data(train_list, eval_list, test_list) # tests if export_clean_data can take in 3 lists and export them as JSON files

    assert export_json_mock.call_count == 3 # assert that the export_json_mock function was called 3 times
    assert logger.info.called # assert that the logger.info function was called

if __name__ == "__main__":
    pytest.main()

