# -*- coding: utf-8 -*-
import os
# from .data_utils import load_params
import glob
import fasttext
import json
from tqdm import tqdm
import re


_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_full_path(path):
    return os.path.join(_ROOT, "../../../", path)


def list_annotated_datasets():
    """
    A function which lists the available annotated training datasets.
    :return: list of categories and file paths of available annotated datasets.
    """
    categories, filenames = [], []
    params = load_params()
    inputfilepath = get_full_path(params['data']['path_to_annotated_data'] + "/*.jsonl")
    filenames = glob.glob(inputfilepath)
    for fn in filenames:
        head, tail = os.path.split(fn)
        categories.append(tail.split('_sentences')[0])
    return categories, filenames


def prodigy_to_fasttext(inputfilepath):
    """
    A function which converts datasets from the Prodigy format to the FastText format.
    :input: path to annotated dataset that is in the Prodigy format
    :return: FastText-formatted dataset.
    """
    inputData = []
    ignoresentences = ["ignore"]
    # For every record in the prodigy formatted file, pick only the label, answer and the sentence
    with open(inputfilepath, 'r', encoding='utf-8') as f:
        for record in f:
            data = json.loads(record)
            if data.get('answer') not in ignoresentences:
                text = "__label__" + data.get('label') + "__" + data.get('answer') + " " + data.get('text')
                inputData.append(text)
    return inputData


def split_annotated_data(inputData,category):
    """
    A function which splits the given datasets into training, validation and test datasets.
    :input:
        inputData: the dataset to split
        category: the classifier to which the dataset belongs
    :return:
    """
    params = load_params()
    train_filepath = get_full_path(params['data']['path_to_fasttext_training_data'] + category + "_ft_train.txt")
    test_filepath = get_full_path(params['data']['path_to_fasttext_test_data'] + category + "_ft_test.txt")
    valid_filepath = get_full_path(params['data']['path_to_fasttext_validation_data'] + category + "_ft_valid.txt")
    [train,test,valid] = params['fasttext']['train_test_valid_ratio']
    inputLength = len(inputData)
    tr_end = round(train*inputLength)
    te_end = tr_end + round(test*inputLength)

    # Save as text files
    with open(train_filepath, 'w', encoding="utf-8") as f:
        for record in inputData[0:tr_end]:
            f.write("%s\n" % record)
    with open(test_filepath, 'w', encoding="utf-8") as f:
        for record in inputData[tr_end:te_end]:
            f.write("%s\n" % record)
    with open(valid_filepath, 'w', encoding="utf-8") as f:
        for record in inputData[te_end:]:
            f.write("%s\n" % record)
    return


def train_individual_classifier(category):
    """
    A function which trains an individual classifier.
    :input:
        category: the classifier to be trained
    :return:
    """
    params = load_params()
    inputfilepath = get_full_path(params['data']['path_to_fasttext_training_data'] + category + "_ft_train.txt")
    modelfilepath = get_full_path(params['data']['path_to_fasttext_individual_models'] + category + "_ft.bin")
    pretrainedvectorspath = get_full_path(params['data']['path_to_fasttext_pretrainedvectors'] + "wiki-news-300d-1M.vec")
    validationfilepath = get_full_path(params['data']['path_to_fasttext_validation_data'] + category + "_ft_valid.txt")

    lr = params['fasttext']['lr']
    epoch = params['fasttext']['epoch']
    wordNgrams = params['fasttext']['wordNgrams']
    minn = params['fasttext']['minn']
    thread = params['fasttext']['thread']
    dim = params['fasttext']['dim']
    loss = params['fasttext']['loss']
    bucket = params['fasttext']['bucket']

    model = fasttext.train_supervised(
        input=inputfilepath,
        lr=lr,
        epoch=epoch,
        wordNgrams=wordNgrams,
        minn=minn,
        thread=thread,
        dim=dim,
        loss=loss,
        bucket=bucket,
        verbose=2,
        pretrainedVectors=pretrainedvectorspath,
        autotuneValidationFile=validationfilepath
    )

    model.save_model(modelfilepath)
    return


def demo_train():
    """
    A function which trains classifiers for all of the available categories.
    """
    inputData = []
    categories, filenames = list_annotated_datasets()
    for i, v in enumerate(tqdm(categories)):
        inputData = prodigy_to_fasttext(filenames[i])
        j=0
        all_records = []
        for record in inputData:
            if re.search("^__label__", record):
                all_records.append(record)
                all_records[-1] = all_records[-1].replace("\n", "")
            elif not re.search("^__label__", record):
                if j > 0:
                    all_records[-1] = all_records[-1] + record
                    all_records[-1] = all_records[-1].replace("\n", "")
            j += 1
        inputData = all_records
        split_annotated_data(inputData, categories[i])
        train_individual_classifier(categories[i])


if __name__ == '__main__':
    demo_train()

