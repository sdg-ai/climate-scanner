# Imports
from .data_utils import DataUtils

class ClimateClassifier():
    def __init__(self):
        # TODO: add initialise logger
        # log.info("Initializing classifier object")
        pass

    def train(self, final__data):
        """
        1. split data to training and test set
        2. fit the model on training data
        3. parameter tuning
        """
        pass

    def test(self):
        """
        A function to test the classifier.
        :return:
        """
        pass
    
    def predict(self, test__data):
        """
        predict classes (relevant/not relevant) for the test data
        """
        output = {"climate_scanner_prob":1}
        return output
    
    def evaluate(self, predictions):
        """
        evaluate the predictions
        """
        pass

class Doc2Climate:

    def __init__(self):
        self.climate_classifier = ClimateClassifier()
        self.utils = DataUtils()

    def get_climate_class(self,input_dict=None):
        text = input_dict["doc"]
        processed_data = self.utils.pre_processing(text)
        cleaned_data = self.utils.data_cleaning(processed_data)
        output = self.climate_classifier.predict(cleaned_data)
        input_dict.update(output)
        return input_dict
