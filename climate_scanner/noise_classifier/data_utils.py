import os
import yaml


_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_params():
    """
    A function which loads the configuration parameters.
    :return: configuration parameters.
    """
    with open(os.path.join(_ROOT, "data", "config.yml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return params




