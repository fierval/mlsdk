import sys
import os, glob

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    __package__ = "iuml"

from .tools import train_utils

train_utils.create_trainer("Unet", r'Y:\nuclei\1\cnn\training')

