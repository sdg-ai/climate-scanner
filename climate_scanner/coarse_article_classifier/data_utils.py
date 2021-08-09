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
    
