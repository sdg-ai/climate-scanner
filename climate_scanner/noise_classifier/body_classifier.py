# -*- coding: utf-8 -*-

import yaml
# from climate_scanner.noise_classifier.data_utils import load_params
from data_utils import load_params
import os
import spacy


_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_full_path(path):
    return os.path.join(_ROOT, path)


# loading config params
with open(os.path.join(_ROOT, 'data', "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

params = load_params()


class BodyNonbodyClassifier:
    """A class to define functions to train/test/predict and evaluate the noise classifier.
    """

    def __init__(self):
        try:
            self.model = spacy.load("en_noise_binary_model")
            print(f"\nNoise model successfully loaded.")
        except:
            print("\nFailed to load model.")

    def predict(self, text):
        """
        A function which predicts whether the text is noise.
        :inputs:
            text: text to predict.
        :return:
            is_body: whether the text is body or non-body (noise).
        """

        threshold = params['predict']['threshold']
        print(f"Threshold for prediction is set to {threshold}.")

        doc = self.model(text)
        prediction = doc.cats
        for item in prediction:
            predict_label = item
            predict_prob = prediction[predict_label]

        output = "BODY" if predict_prob > threshold else "NOISE"
        print(f"\ntext = {text} \nPrediction output = {output} (Likelihood of being {predict_label} is {predict_prob})")

        return output


def main():
    """
    pipeline to read data/process data /train/ test /evaluate
    :return:
    """
    x = BodyNonbodyClassifier()
    # text = "2014b"
    text = "Lithium ion battery life has greatly improved electric transport."
    output = x.predict(text)
    print(output)


if __name__ == "__main__":
    main()
