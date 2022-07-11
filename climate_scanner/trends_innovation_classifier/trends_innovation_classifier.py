# -*- coding: utf-8 -*-
import os
import yaml
from climate_scanner.trends_innovation_classifier.data_utils import doc_to_sentence,data_processing, load_params
import glob
import os
import spacy
from operator import itemgetter

_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_full_path(path):
    return os.path.join(_ROOT, path)

# loading config params
with open(os.path.join(_ROOT, 'data', "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

params = load_params()

class TrendsInnovationClassifier:
    """A class to define functions to train/test/predict and evalute the classifier.
    """

    def __init__(self):
        self.categories, self.model_names = self._list_all_classifiers()
        self.models = self.load_all_classifiers(self.model_names)

    def _list_all_classifiers(self):
        """
        A function which lists the available classifiers.
        :return: list of categories of trends and innovations for which there are classifiers.
        """
        categories = []
        model_names = []
        inputfilepath = get_full_path(params['data']['path_to_prodigy_models'])
        print(inputfilepath)
        # filenames = glob.glob(inputfilepath)
        # my_list = os.listdir('My_directory')
        filenames = [x[0] for x in os.walk(inputfilepath)]
        print(filenames)

        for fn in filenames:
            head, tail = os.path.split(fn)
            print(head, tail)
            categories.append(tail)
        print(categories, model_names)
        print(len(categories), len(model_names))
        filtered_filenames = []
        filtered_categories = []
        for filepath, category in zip(filenames, categories):
            if '_model' in category:
                filtered_filenames.append(filepath)
                filtered_categories.append(category)

        print(filtered_filenames, filtered_categories)

        return filtered_categories, filtered_filenames

    def load_all_classifiers(self, model_names):
        """
        A function which loads the available classifiers.
        """

        all_classifiers = []

        for model in model_names:
            print(model)
            try:
                model_best = os.path.join(model,'model-best')
                all_classifiers.append(spacy.load(model_best))
            except Exception as e:
                print(e)

        return all_classifiers

    def predict(self, text):
        """
        A function which predicts the output from all available classifiers.
        :inputs:
            text: text to predict.
        :return:
            most_likely_trends: list of most likely trends and innovations.
        """
        all_labels, all_probs, most_likely_trends = [], [], []
        threshold = params['trends_demo']['threshold']
        count = params['trends_demo']['count']
        print(count, threshold)
        print(len(self.models))

        for nlp in self.models:
            doc = nlp(text)
            prediction = doc.cats
            for item in prediction:
                predict_label = item
                predict_prob = prediction[predict_label]
            print("\nmodel: {} \t predict_label {} \t predict_prob {}: ".format(self.categories[self.models.index(nlp)],
                                                                                predict_label, predict_prob))
            all_labels.append(predict_label)
            all_probs.append(predict_prob)

        most_likely_trends = [[all_labels[i], all_probs[i]] for i, v in enumerate(all_probs) if v > threshold]
        most_likely_trends = sorted(most_likely_trends, key=itemgetter(1), reverse=True)
        print("most likely trends contains: ", most_likely_trends)
        try:
            most_likely_trends = [most_likely_trends[i] for i in range(count)]
        except:
            print("{} trends match the requested threshold.".format(len(most_likely_trends)))

        return most_likely_trends

    def demo_return(self, text):
        """
        A demo function to predict the most likely trends and innovations.
        :inputs: text to classify.
        :return: dictionary containing the classified trends and innovations.
        """

        most_likely_trends = self.predict(text)


        output = {"string_prediction": [most_likely_trends[i][0] for i in range(len(most_likely_trends))],
                  "string_prob": [most_likely_trends[i][1] for i in range(len(most_likely_trends))]}

        return output

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


    def eval(self):
        """
        A function to evaluate the classifier.
        :return:
        """
        pass

class Doc2Trends:

    def __init__(self):
        self.tic = TrendsInnovationClassifier()


    def coordinator_pipeline(self,input_dict=None,threshold=None):
        id_dict = dict()
        id_dict["ID"] = input_dict["ID"]
        if input_dict["climate_scanner_prob"] >= threshold:
            text = input_dict["doc"]
            split_sentences = doc_to_sentence(text)
            all_predictions = []
            for dic in split_sentences:
                sentence = data_processing(dic["text"])
                output = self.tic.predict(sentence)
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
    x = TrendsInnovationClassifier()
    text = "Lithium ion battery life has greatly improved electric transport. " \
           "Electric motors can now run for hundreds of kilometers without needing to refuel. " \
           "Car manufacturing is improving year on year"
    output = x.demo_return(text)
    print(output)


if __name__ == "__main__":
    main()
