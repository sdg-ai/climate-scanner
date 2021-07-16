# -*- coding: utf-8 -*-
import os
import yaml
from data_utils import doc_to_sentence,data_processing

_ROOT = os.path.abspath(os.path.dirname(__file__))

# loading config params
with open(os.path.join(_ROOT,"config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

class TrendsInnovationClassifier:
    """A class to define functions to train/test/predict and evalute the classifier.
    """

    def __init__(self):
        Doc2Trends.__init__(self)

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

    def predict(self,text):
        """
        A function to predict using the classifier.
        :return:
        """
        output = {"string_prediction":["building","3-d printing"],"string_prob":[0.9,0.5]}
        return output

    def eval(self):
        """
        A function to evaluate the classifier.
        :return:
        """
        pass

class Doc2Trends:

    def __init__(self,input_dict=None,threshold=None):
        self.input = input_dict
        self.threshold = threshold

    def coordinator_pipeline(self):
        id_dict = dict()
        id_dict["ID"] = self.input["ID"]
        if self.input["climate_scanner_prob"] >= self.threshold:
            text = self.input["doc"]
            split_sentences = doc_to_sentence(text)
            all_predictions = []
            for dic in split_sentences:
                sentence = data_processing(dic["text"])
                tic = TrendsInnovationClassifier()
                output = tic.predict(sentence)
                dic.update(output)
                all_predictions.append(dic)
        else:
            all_predictions = []
        return [id_dict] + all_predictions


def main():
    """
    pipeline to read data/process data /train/ test /evaluate
    :return:
    """
    pass


if __name__ == "__main__":
    main()
