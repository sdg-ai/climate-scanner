import os
import yaml
import nltk
import math

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


def doc_to_multisentence(text:str,num_sentences:int):
    
    """
        Sentence splitter and string indices generator. Splits the text into groups of sentences and generates string indices.
        Arguments: Str - Free Text Ex: Title / Document / Article etc.
                 : Int - Number of sentences that are grouped together.
        :return: Str - List of dictionaries having string indices and text.
        """
    
    sentences = nltk.sent_tokenize(text)
    offset = 0
    sent_dict = dict()
    sent_dict["string_indices"] = [0]
    sent_dict["text"] = ""
    split_sentences = []
    j = 0
    
    for k in range(math.ceil(len(sentences)/num_sentences)):
        for line in sentences[j:min(j+num_sentences,len(sentences))]:
            offset = text.find(line,offset)
            sent_dict.get("string_indices").append(offset+len(line)+1)
            sent_dict["text"] = sent_dict["text"] + ' ' + line
        sent_dict["text"] = sent_dict["text"][1:]
        split_sentences.append(sent_dict.copy()) 
        sent_dict = dict()
        sent_dict["string_indices"] = [offset+len(line)+2]
        sent_dict["text"] = ""
        j += num_sentences
    split_sentences[-1]["string_indices"][-1] = offset+len(line)
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
