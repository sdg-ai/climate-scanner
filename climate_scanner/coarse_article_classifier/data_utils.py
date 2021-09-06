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
        text = self._remove_hyperlinks(text)
        text = self._remove_hashtags(text)
        text = self._remove_html_tags(text)
        text = self._remove_numbers(text)
        text = self._encode_unknown(text)
        text = self._clean_eol_tabs(text)
        text = self._clean_punctuation_no_accent(text)
        text = self._clean_stopwords(text)
        text = text.replace(r'&lt;p&gt;', '')
        text = text.replace(r'&amp;apos;','')
        text = text.replace(r'<.*?>', '')
        text = text.replace(r'http\S+', '')
        return text

    def data_cleaning(self, text):
        """
        A function to numerify the data and prepare it for the next stages of modeling
        while ensuring least possible loss of information
        """
        pass

    def _remove_hyperlinks(self, text):
        """ This function removes hyperlinks from texts
            inputs:
            - text """ 
        text = text.apply(lambda x: re.sub(r"http\S|www\.\S++", " ", x))
        return text
    
    def _remove_hashtags(self, text):
        """ This function removes hashtags
            inputs:
            - text """
        text = text.apply(lambda x: re.sub(r"#\w+", " ", x))
        return text

    def _remove_html_tags(self, text):
        """ This function removes html tags from texts
            inputs:
            - text """
        text = text.apply(lambda x: re.sub(r"<.*?>", " ", x))
        return text


    def _remove_numbers(self, text):
        """ This function removes numbers from a text
            inputs:
            - text """
        text = text.apply(lambda x: re.sub(r"\d+", " ", x))
        return text

    
    def _encode_unknown(self, text):
        """ This function encodes special caracters """
        text = text.apply(lambda x: unicodedata.normalize("NFD", x).encode('ascii', 'ignore').decode("utf-8"))
        return text


    def _clean_eol_tabs(self, text):
        """ text lowercase
            removes \n
            removes \t
            removes \r """
        text = text.str.lower()
        text = text.apply(lambda x: x.replace("\n", " "))
        text = text.apply(lambda x: x.replace("\r", " "))
        text = text.apply(lambda x: x.replace("\t", " "))
        return text


    def _clean_punctuation_no_accent(self, text):
        """ This function removes punctuation and accented characters from texts in a dataframe 
            To be appplied to languages that have no accents, ex: english 
        """
        text = text.apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
        return text
    

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

    def _clean_stopwords(self, text):
        """ This function removes stopwords """
        text = text.apply(lambda x: self._remove_stop_words(x))
        return text
