# -*- coding: utf-8 -*-

import os
from keras.preprocessing import text, sequence
from keras import Model 

#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_data(path):
	return os.path.join(_ROOT, 'data', path)

class classifier(Model):
    # child class inherited from "keras.Model" defining the sentiment classifier model
    # methods:
    #   call     -> Method defining the forward pass in the classifier model
    #   evaluate -> Method defining model evaluation metrics a.k.a confusion matrix, precision, recall, ROC,...etc. 

    
    def __init__(self):
        super(classifier, self).__init__()
        # example of model definition:
        # self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        # self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        # self.dropout = tf.keras.layers.Dropout(0.5)
        pass

    def call(self,inputs,training=False):
        # Method defining the forward pass in the classifier model (called internally during forward pass in compile method)
        # Args:
        #   inputs   ->  ....
        #   training ->  (bool) used to specify different behavior during training phase training 
        # returns:
        #   lastTensor -> tensor representing last layer in the model    
        
        ### Example:
        # x = self.dense1(inputs)
        # if training:
        #     x = self.dropout(x, training=training)
        # lastTensor = self.dense2(x)
        # return lastTensor
        pass

    def evaluate(self,**kwargs):
        # Method defining model evaluation metrics a.k.a confusion matrix, precision, recall, ROC,...etc. 
        # Args:
        #     .....
        # Returns:
        #     .....
        pass


class data_preprocessing:
    # class defining data pre-processing methods and process
    # methods:
        # method1:...
        pass


    

