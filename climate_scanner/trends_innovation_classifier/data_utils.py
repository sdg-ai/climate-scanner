import os
import yaml

_ROOT = os.path.abspath(os.path.dirname(__file__))

# loading config params
with open(str(_ROOT / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


def load_data():
    """
    A Data loader function which reads data from location.
    :return: input vectors, labels, vocabulary.
    """
    pass


def data_processing():
    """
    Preprocessing data
    :return:
    """
    pass


def batch_iter():
    """
    A helper function to convert data into tensors and pass batches for training.
    :return:
    """
    pass
