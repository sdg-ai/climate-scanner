import re
import unicodedata

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


class DataUtils:

    def load_data():
        """
        A Data loader function which reads data from location.
        :return: input vectors, labels, vocabulary.
        """
        pass

    def pre_processing(self, text):
        """
        An function to clean the data if it contains any special characters, http elements, unicode characters etc.
        Arguments: Str - Free Text Ex: Title / Document / Article etc.
        :return: Str - Cleaned text
        """

        text = text.lower()
        # remove hyperlinks 
        text = text.apply(lambda x: re.sub(r"http\S|www\.\S++", " ", x)) 
        # remove hashtags
        text = text.apply(lambda x: re.sub(r"#\w+", " ", x))
        # remove html_tags
        text = text.apply(lambda x: re.sub(r"<.*?>", " ", x))
        # remove numbers
        text = text.apply(lambda x: re.sub(r"\d+", " ", x))
        # encode unknown
        text = text.apply(lambda x: unicodedata.normalize("NFD", x).encode('ascii', 'ignore').decode("utf-8"))
        # clean end-of-line tabs
        text = text.str.lower()
        text = text.apply(lambda x: x.replace("\n", " "))
        text = text.apply(lambda x: x.replace("\r", " "))
        text = text.apply(lambda x: x.replace("\t", " "))
        # clean punctuations
        text = text.apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
        # remove stopwords
        text = text.apply(lambda x: self._remove_stop_words(x))

        text = text.replace(r'&lt;p&gt;', '')
        text = text.replace(r'&amp;apos;','')

        return text

    def data_cleaning(self, text):
        """
        A function to numerify the data and prepare it for the next stages of modeling
        while ensuring least possible loss of information
        """
        pass

    def _remove_stop_words(self, text, stopwords=set(stopwords.words('english'))):
        """ This function removes stop words from a text
            inputs:
            - stopword list
            - text """

        # prepare new text
        text_splitted = text.split(" ")
        text_new = list()
        
        # stop words updated
        # stopwords = stopwords.union({"amp", "grocery store", "covid", "supermarket", "people", "grocery", "store", "price", "time", "consumer"})

        for word in text_splitted:
            if word not in stopwords:
                text_new.append(word)
        return " ".join(text_new)
