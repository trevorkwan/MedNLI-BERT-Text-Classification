import os
import logging
from transformers.trainer_utils import get_last_checkpoint # returns the path to the last saved checkpoint file during training (useful if training is interrupted)

# need to set logger in helper function scripts
logger = logging.getLogger(__name__)

def detect_last_checkpoint(training_args):
    """
    Detects the last checkpoint if available in the output directory.

    Args:
        training_args (MyTrainingArguments): An instance of MyTrainingArguments containing the training arguments.

    Returns:
        str or None: The last checkpoint path if available, else None.
    """
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint