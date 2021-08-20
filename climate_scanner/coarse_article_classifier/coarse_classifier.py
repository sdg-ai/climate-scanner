# Imports
from data_utils import DataUtils

class ClimateClassifier():
    def __init__(self, )
        log.info("Initializing classifier object")

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

    def get_climate_class(self,input_dict=None):        
        text = input_dict["text"]
        all_predictions = []
        for dic in text:
            processed_data = DataUtils.pre_processing(dic["text"])
            cleaned_data = DataUtils.data_cleaning(processed_data)
            output = self.climate_classifier.predict(cleaned_data)
            dic.update(output)
            all_predictions.append(dic)

        return input_dict["id"] + all_predictions