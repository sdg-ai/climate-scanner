# -*- coding: utf-8 -*-
import os
import yaml

_ROOT = os.path.abspath(os.path.dirname(__file__))

# loading config params
with open(str(_ROOT / "config.yml")) as f:
    params = yaml.load(f, Loader = yaml.FullLoader)


class TrendsInnovationClassifier:
    """A class to define functions to train/test/predict and evalute the classifier.
    """

    def __init__(self):
        pass

    def train(self):
        """
        A function to train the classifier.
        :return:
        """
        pass

    def test(self):
        """
        A function to test the classifier.
        :return:
        """
        pass

    def predict(self):
        """
        A function to predict using the classifier.
        :return:
        """
        pass

    def eval(self):
        """
        A function to evaluate the classifier.
        :return:
        """
        pass


def main():
    """
    pipeline to read data/process data /train/ test /evaluate
    :return:
    """
    pass


if __name__ == "__main__":
    main()
