import os
import yaml
import nltk

_ROOT = os.path.abspath(os.path.dirname(__file__))

# loading config params
with open(os.path.join(_ROOT,"config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


def load_data():
    """
    A Data loader function which reads data from location.
    :return: input vectors, labels, vocabulary.
    """
    pass

def doc_to_sentence(text:str):
    """
        Sentence splitter and string indices generator. Splits the text into sentences and generates string indices.
        Arguments: Str - Free Text Ex: Title / Document / Article etc.
        :return: Str - List of dictionaries having string indices and text.
        """
    sentences = nltk.sent_tokenize(text)
    offset = 0
    sent_dict = dict()
    split_sentences = []
    for line in sentences:
        offset = text.find(line,offset)
        sent_dict["string_indices"] = (offset,offset+len(line))
        sent_dict["text"] = line
        split_sentences.append(sent_dict.copy())
    return split_sentences


def data_processing(text:str) -> str:
    """
    An function to clean the data if it contains any special characters, http elements, unicode characters etc.
    Arguments: Str - Free Text Ex: Title / Document / Article etc.
    :return: Str - Cleaned text
    """
    text = text.lower()
    text = text.replace(r'&lt;p&gt;', '')
    text = text.replace(r'&amp;apos;','')
    text = text.replace(r'<.*?>', '')
    text = text.replace(r'http\S+', '')
    return text




def batch_iter():
    """
    A helper function to convert data into tensors and pass batches for training.
    :return:
    """
    pass
