import os
import yaml
import nltk
import json
import math

_ROOT = os.path.abspath(os.path.dirname(__file__))


# # loading config params
# with open(os.path.join(_ROOT, "data", "config.yml")) as f:
#     params = yaml.load(f, Loader=yaml.FullLoader)


def load_params():
    """
    A function which loads the configuration parameters.
    :return: configuration parameters.
    """
    with open(os.path.join(_ROOT, "data", "config.yaml")) as f:   
        params = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return params


# def load_data():
#     """
#     A Data loader function which reads data from location.
#     :return: input vectors, labels, vocabulary.
#     """
#     pass


def load_data():
    """
    A data loader function which reads in the raw training dataset.
    :return: input vectors, labels, vocabulary.
    """
    
    inputData, inputCategories, inputText = [], [], []
    params = load_params()
    inputFile = os.path.join(params['data']['path_to_trainingData'] , params['pre_training']['input_fileName'])
    
    with open(inputFile, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            inputData.append(data)
    
    inputCategories = [set(i.get('category') for i in inputData)]
    inputText = [i.get('text') for i in inputData]
    
    return inputData, inputCategories, inputText


def pull_categoryData(inputData,category:str):
    outputData = []
    for i,d in enumerate(inputData):
        if category in d['category']:
            outputData.append(d.copy())
    return outputData if outputData else print("Incorrect category name!")


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


def build_trainingData(inputData):
    outputText = {}
    outputData = []
    params = load_params()
    num_sentences = params['pre_training']['num_sentences']
    
    for d in inputData:
        count, sent_id = 0, 0
        id_array, pos_array, text_array = [], [], []
        inputText = d.get("text") 
        outputText["id"] = d.get("id")
        outputText["title"] = d.get("title")
        splitText = doc_to_sentence(inputText)
        for st in splitText:
            sent_id = count
            sent_pos = st.get("string_indices")
            sent_text = st.get("text")
            id_array.append(sent_id)
            pos_array.append(sent_pos)
            text_array.append(sent_text)
            count += 1
        count = 0
        quotient = math.ceil(len(text_array)/num_sentences)
        remainder = math.ceil(len(text_array)%num_sentences)
        ta_range = quotient
        for ta in range(ta_range):
            if remainder == 0 or (remainder > 0 and ta < quotient - 1):
                k_range = num_sentences
            elif (remainder > 0 and ta == quotient - 1):
                k_range = remainder
            elif (len(text_array) == 1):
                k_range = 1
            for k in range(k_range):
                if len(text_array) == 1:
                    outputText['prev_id'] = []
                    outputText['prev_pos'] = []
                    outputText['prev_text'] = ""
                    outputText['curr_id'] = id_array[count]
                    outputText['curr_pos'] = pos_array[count]
                    outputText['curr_text'] = text_array[count]
                    outputText['next_id'] = []
                    outputText['next_pos'] = []
                    outputText['next_text'] = ""
                elif count == 0:
                    outputText['prev_id'] = []
                    outputText['prev_pos'] = []
                    outputText['prev_text'] = ""
                    outputText['curr_id'] = id_array[count]
                    outputText['curr_pos'] = pos_array[count]
                    outputText['curr_text'] = text_array[count]
                    outputText['next_id'] = id_array[count+1]
                    outputText['next_pos'] = pos_array[count+1]
                    outputText['next_text'] = text_array[count+1]
                elif count == len(text_array)-1:
                    outputText['prev_id'] = id_array[count-1]
                    outputText['prev_pos'] = pos_array[count-1]
                    outputText['prev_text'] = text_array[count-1]
                    outputText['curr_id'] = id_array[count]
                    outputText['curr_pos'] = pos_array[count]
                    outputText['curr_text'] = text_array[count]
                    outputText['next_id'] = []
                    outputText['next_pos'] = []
                    outputText['next_text'] = ""
                else:
                    outputText['prev_id'] = id_array[count-1]
                    outputText['prev_pos'] = pos_array[count-1]
                    outputText['prev_text'] = text_array[count-1]
                    outputText['curr_id'] = id_array[count]
                    outputText['curr_pos'] = pos_array[count]
                    outputText['curr_text'] = text_array[count]
                    outputText['next_id'] = id_array[count+1]
                    outputText['next_pos'] = pos_array[count+1]
                    outputText['next_text'] = text_array[count+1]
                outputText['category'] = d.get('category')
                outputText['climate_scanner'] = d.get('climate_scanner')
                outputData.append(outputText.copy())
                count += 1
    return outputData
    

def format_outputData(outputData:list):
    """
    A function which formats the resulting training data for Prodigy.
    """
    
    outputText,metaData = {},{}
    prodigyData = []
    for d in outputData:
        outputText={}
        outputText['doc_id'] = d.get('id')
        outputText['sent_id']=d.get('curr_id')
        if d.get('prev_pos')==[]:
            outputText['sent_start_pos']=[] 
        else: 
            outputText['sent_start_pos']=d.get('prev_pos')[0] 
        if d.get('next_pos')==[]: 
            outputText['sent_end_pos']=[] 
        else:
            outputText['sent_end_pos']=d.get('next_pos')[1]
        outputText['title'] = d.get('title')
        outputText['category'] = d.get('category')
        outputText['climate_scanner'] = d.get('climate_scanner')
        metaData["meta"]=outputText
        metaData["text"]=d.get('prev_text')+' '+d.get('curr_text')+' '+d.get('next_text')
        prodigyData.append(metaData.copy())
    return prodigyData

            
def save_outputData(outputData:list):
    """
    A function which writes the resulting training data from a given list of dictionaries to file.
    """
    
    params = load_params()
    outputFile = os.path.join(params['data']['path_to_trainingData'], params['pre_training']['output_fileName'])
    
    with open(outputFile, 'w', encoding='utf-8') as f:
        for line in outputData:
            f.write(json.dumps(line, sort_keys=True))
            f.write("\n")
            

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