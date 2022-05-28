# -*- coding: utf-8 -*-
import os
import yaml
from .data_utils import load_params

import glob
import fasttext
from operator import itemgetter

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_full_path(path):
    return os.path.join(_ROOT, "../../../", path)


def list_all_classifiers():
    """
    A function which lists the available classifiers.
    :return: list of categories of trends and innovations for which there are classifiers.
    """
    categories = []
    params = load_params()
    inputfilepath = get_full_path(params['data']['path_to_fasttext_individual_models'] + "/*.bin")
    filenames = glob.glob(inputfilepath)

    for fn in filenames:
        head, tail = os.path.split(fn)
        categories.append(tail.split('_ft')[0])

    return categories


def load_all_classifiers(categories):
    """
    A function which loads the available classifiers.
    """
    all_classifiers = []
    params = load_params()

    for imodel, vmodel in enumerate(categories):
        modelfilepath = get_full_path(params['data']['path_to_fasttext_individual_models'] + vmodel + "_ft.bin")
        all_classifiers.append(fasttext.load_model(modelfilepath))

    return all_classifiers


def predict_against_all_individual_classifiers(text):
    """
    A function which predicts the output from all available classifiers.
    :inputs:
        text: text to predict.
    :return:
        most_likely_trends: list of most likely trends and innovations.
    """
    all_labels, all_probs, most_likely_trends = [], [], []
    params = load_params()
    threshold = params['trends_demo']['threshold']
    count = params['trends_demo']['count']
    categories = list_all_classifiers()
    all_classifiers = load_all_classifiers(categories)

    for model in all_classifiers:
        predict_label, predict_prob = model.predict(text)
        predict_label = predict_label[0][9:-8]
        predict_prob = predict_prob[0]
        all_labels.append(predict_label)
        all_probs.append(predict_prob)

    most_likely_trends = [[all_labels[i], all_probs[i]] for i, v in enumerate(all_probs) if v > threshold]
    most_likely_trends = sorted(most_likely_trends, key=itemgetter(1), reverse=True)
    try:
        most_likely_trends = [most_likely_trends[i] for i in range(count)]
    except:
        print("{} trends match the requested threshold.".format(len(most_likely_trends)))

    return most_likely_trends


def demo_classifier(text):
    """
    A demo function to predict the most likely trends and innovations.
    :inputs: text to classify.
    :return: dictionary containing the classified trends and innovations.
    """
    most_likely_trends = predict_against_all_individual_classifiers(text)

    output = {"string_prediction": [most_likely_trends[i][0] for i in range(len(most_likely_trends))],
              "string_prob": [most_likely_trends[i][1] for i in range(len(most_likely_trends))]}

    return output


if __name__ == '__main__':
    text = "Lithium ion battery life has greatly improved electric transport."
    output = demo_classifier(text)
    print("\nOutput: ", output)
