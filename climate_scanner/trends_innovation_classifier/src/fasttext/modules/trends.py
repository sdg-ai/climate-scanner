# -*- coding: utf-8 -*-
import os
import yaml
from .data_utils import load_params

import glob
import fasttext
import operator
from operator import itemgetter



# _ROOT = os.path.abspath(os.path.dirname(__file__))

def list_all_classifiers():
    """
    A function which lists the available classifiers.
    :return: list of categories of trends and innovations for which there are classifiers.
    """
    
    params = load_params()
    inputfilepath = params['data']['path_to_annotated_data']
    categories = []
    inputfilename = inputfilepath + "/*.jsonl"
    filenames = glob.glob(inputfilename)
    
    for fn in filenames:
        head, tail = os.path.split(fn) 
        categories.append(tail.split('_sentences')[0])    
    
    return categories


def predict_using_individual_classifier(text, model_classifier):
    """
    A function which predicts the output from an individual classifier.
    :inputs: 
        text: text to predict.
        model_classifier: classifier to use for the prediction.
    :return: 
        predict_prob: prediction confidence.
        predict_label: predicted class.
    """
    
    params = load_params()
    modelfilepath = params['data']['path_to_fasttext_individual_models'] + model_classifier + "_ft.bin"
    predict_label, predict_prob = [], []
    
    model = fasttext.load_model(modelfilepath)
    
    predict_label, predict_prob = model.predict(text)  
    predict_label = predict_label[0][9:-8]
    predict_prob = predict_prob[0]
    
    return predict_prob, predict_label


def predict_against_all_individual_classifiers(text, threshold, count):
    """
    A function which predicts the output from all available classifiers.
    :inputs: 
        text: text to predict.
        threshold: probability threshold (from 0 to 1) to filter out entries
        count: number of output classes to provide.
    :return: 
        most_likely_trends: list of most likely trends and innovations.
    """
    
    all_labels, all_probs, most_likely_trends = [], [], []
    categories = list_all_classifiers() 
    
    for imodel,vmodel in enumerate(categories):
        predict_prob, predict_label = predict_using_individual_classifier(text,categories[imodel])
        all_labels.append(predict_label) 
        all_probs.append(predict_prob)
    
    most_likely_trends = [[all_labels[i],all_probs[i]] for i,v in enumerate(all_probs) if v > threshold]
    most_likely_trends = sorted(most_likely_trends, key=itemgetter(1), reverse=True)
    most_likely_trends = [most_likely_trends[i] for i in range(count)]
    
    return most_likely_trends


def demo_classifier(text):
    """
    A demo function to predict the most likely trends and innnovations.
    :inputs: text to classify.
    :return: dictionary containing the classified trends and innovations.
    """
    
    threshold = 0.5
    count = 1
    
    most_likely_trends = predict_against_all_individual_classifiers(text, threshold, count)
    
    output = {"string_prediction":[most_likely_trends[i][0] for i in range(len(most_likely_trends))],
              "string_prob":[most_likely_trends[i][1] for i in range(len(most_likely_trends))]}

    return output

