# MedNLI Text Classification Project

This repository contains the code for training and evaluating a text classification model using the MedNLI dataset. The model is trained using the BERT architecture and fine-tuned on the specific dataset. The project utilizes a Makefile to simplify the execution of various tasks such as data cleaning, hyperparameter optimization, training, evaluation, and prediction.

## Prerequisites

- Python 3.6 or later
- Install the required packages by running:

```bash
pip install -r requirements.txt
```

## Using the Makefile

The Makefile contains several targets that correspond to different stages of the project. You can run each target by executing the `make <target_name>` command in the terminal.

### Targets

- `clean_data`: This target cleans and preprocesses the raw dataset, generating train, validation, and test files in a clean format.
- `optimize`: This target performs hyperparameter optimization using Optuna, searching for the best hyperparameters for the model.
- `train`: This target fine-tunes the BERT model using the cleaned dataset and optimized (or default) hyperparameters.
- `eval`: This target evaluates the fine-tuned model on the test dataset and outputs various evaluation metrics.
- `predict`: This target generates predictions for the test dataset using the fine-tuned model.

### Running the Makefile

To run the entire pipeline from data cleaning to prediction, simply execute the following command:
```bash
make all
```

To run a specific target, execute the command:
```bash
make <target_name>
```

For example, to run the train target, use the command:
```bash
make train
```

### Customizing the Makefile
You can customize the Makefile by modifying the parameters passed to each script. For example, you can change the learning rate for the train target by updating the --learning_rate parameter in the Makefile:

```make
train: 
	$(PYTHON) $(TRAIN_SCRIPT) --train_file clean_train_med.json \
                            ...
                            --learning_rate 5e-5 \
                            ...
                            --overwrite_output_dir True
```

After making any changes to the Makefile, simply re-run the corresponding target using the make <target_name> command.

## Troubleshooting
If you encounter any issues or errors during the execution of the Makefile, try the following:

1. Ensure that you have installed all the required packages.
2. Verify that the file paths and parameters in the Makefile are correct.
3. Check the log messages for any additional information about the error.
4. If the error persists, consider re-running the target with the --overwrite_output_dir True flag to overwrite any existing files that might be causing issues.
