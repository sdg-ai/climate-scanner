# -*- coding: utf-8 -*-

import os
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import copy

# tf
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# keras
import keras
import keras.backend as K
from keras import regularizers, optimizers
from keras.layers import Embedding, Dense, Dropout, Input, LSTM, GlobalMaxPool1D
from keras.initializers import Constant
from keras.preprocessing import text, sequence
from keras import Model

import spacy
from sklearn.metrics import classification_report, confusion_matrix

#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
	return os.path.join(_ROOT, 'data', path)


def get_params():
	# Utility method to load parameters:
	# Args:
	# ---
	# Returns:
	# params -> dictionary with parameters
	with open(get_data('config.yaml')) as file:
		params = yaml.load(file, Loader=yaml.FullLoader)
	file.close()
	return params


#############################################################################
#
# 	              SentimentClassifier class
#
#############################################################################

class SentimentClassifier(Model):
	# child class inherited from "keras.Model" defining the sentiment classifier model
	# methods:
	#   call     -> Method defining the forward pass in the classifier model
	#   evaluate -> Method defining model evaluation metrics a.k.a confusion matrix, precision, recall, ROC,...etc.

	def __init__(self, embedding_matrix, params):
		super(SentimentClassifier, self).__init__()

		# path parameters
		self._path_to_logs = params['data']['path_to_logs']
		self._path_to_checkpoints = params['data']['path_to_checkpoints']
		self._path_to_model = params['data']['path_to_model']

		# pre_processing parameters
		self._embedding_matrix = embedding_matrix
		self._vocab_size = params['pre_processing']['vocab_size']
		self._embedding_dim = params['pre_processing']['embedding_dim']

		# Model compile
		self._model_loss = params['compile']['loss']
		self._model_metrics = params['compile']['metrics']

		# Model optimizer
		self._lr = params['optimizer']['lr']
		self._beta_1 = params['optimizer']['beta_1']
		self._beta_2 = params['optimizer']['beta_2']
		self._epsilon = params['optimizer']['epsilon']
		self._amsgrad = params['optimizer']['amsgrad']
		self._opt = keras.optimizers.Adam(learning_rate=self._lr,
										  beta_1=self._beta_1,
										  beta_2=self._beta_2,
										  epsilon=self._epsilon,
										  amsgrad=self._amsgrad)

		# Model Fit
		self._batch_size = params['fit']['batch_size']
		self._epochs = params['fit']['epochs']
		self._verbose = params['fit']['verbose']
		self._validation_split = params['fit']['validation_split']
		self._shuffle = params['fit']['shuffle']
		self._class_weight = params['fit']['class_weight']
		self._sample_weight = params['fit']['sample_weight']
		self._initial_epoch = params['fit']['initial_epoch']
		self._steps_per_epoch = params['fit']['steps_per_epoch']
		self._validation_steps = params['fit']['validation_steps']
		self._validation_batch_size = params['fit']['validation_batch_size']
		self._validation_freq = params['fit']['validation_freq']
		self._max_queue_size = params['fit']['max_queue_size']
		self._workers = params['fit']['workers']
		self._use_multiprocessing = params['fit']['use_multiprocessing']

		# Model Predict:
		self._batch_size_p = params['predict']['batch_size']
		self._verbose_p = params['predict']['verbose']
		self._steps_p = params['predict']['steps']
		self._max_queue_size_p = params['predict']['max_queue_size']
		self._workers_p = params['predict']['workers']
		self._use_multiprocessing_p = params['predict']['use_multiprocessing']

		###############################
		####### MODEL STRUCTURE #######
		###############################

		# Model structure
		self.Embedding = tf.keras.layers.Embedding(self._vocab_size,
												   self._embedding_dim,
												   embeddings_initializer=Constant(embedding_matrix),
												   trainable=False)
		self.LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self._embedding_dim * 2))
		self.dense1 = tf.keras.layers.Dense(24, activation='relu')
		self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

	def call(self, inputs):
		# Method defining the forward pass in the classifier model (called internally during forward pass in compile method)
		# Args:
		#   inputs     ->  np.array() representing tokens
		# returns:
		#   lastTensor -> tensor representing last layer in the model

		x = self.Embedding(inputs)
		x = self.LSTM(x)
		x = self.dense1(x)
		lastTensor = self.dense2(x)
		return lastTensor

	def compile(self):
		# Method to compile the model
		self.compile(optimizer=self._opt,
					 loss=self._model_loss,
					 metrics=self._model_metrics)

	def fit(self, training_data, validation_data=None, call_backs_dir=None):
		# Method to fit the model to current data
		# Args:
		#    training_data   -> tuple(X,y) where X and y are of type numpy.array(). X represents text mapping, y is the label
		#    validation_data -> tuple(X,y) where X and y are of type numpy.array(). X represents text mapping, y is the label, defaults to None
		#    call_back_dir   -> optional, of type str describing name of log directory for callbacks, default is None
		# Returns:
		#    ---
		if call_backs_dir:
			# tensorboard -log files
			log_dir = os.path.join(get_data(self._path_to_logs), call_backs_dir,
								   datetime.now().strftime("%Y%m%d-%H%M%S"))
			tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

			# checkpoints to save model
			checkpoint_dir = os.path.join(get_data(self._path_to_checkpoints), call_backs_dir, '')

			checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
																	 save_freq='epoch',
																	 save_weights_only=True,
																	 monitor='val_accuracy',
																	 save_best_only=True)
			# forming list of callbacks
			call_backs = [tensorboard_callback, checkpoint_callback]
		else:
			call_backs = None

		X_train, y_train = training_data
		self.fit(x=X_train,
				 y=y_train,
				 batch_size=self._batch_size,
				 epochs=self._epochs,
				 verbose=self._verbose,
				 callbacks=call_backs,
				 validation_split=self._validation_split,
				 validation_data=validation_data,
				 shuffle=self._shuffle,
				 class_weight=self._class_weight,
				 sample_weight=self._sample_weight,
				 initial_epoch=self._initial_epoch,
				 steps_per_epoch=self._steps_per_epoch,
				 validation_steps=self._validation_steps,
				 validation_batch_size=self._validation_batch_size,
				 validation_freq=self._validation_freq,
				 max_queue_size=self._max_queue_size,
				 workers=self._workers)

	def predict(self, X, thresh=None):
		# Method to predict sentiment around text
		# Args:
		# X      -> of type numpy.array() representing text mapping used to predict sentiment
		# thresh -> of type float. threshold used around predicted probabilities
		# returns:
		# y_proba -> numpy.array() representing vector of probabilities for each class
		# y_pred  -> numpy.array() representing vector of predicted classes
		y_proba = self.predict(X,
							   batch_size=self._batch_size_p,
							   verbose=self._verbose_p,
							   steps=self._steps_p,
							   max_queue_size=self._max_queue_size_p,
							   workers=self._workers_p,
							   use_multiprocessing=self._use_multiprocessing_p)

		y_pred = (y_proba > thresh).astype(int) if thresh else (y_proba > 0.5).astype(int)
		return y_proba, y_pred

	def evaluate(self, y_true, y_pred):
		# Method to evaluate predictions using classification_report and confusion_matrix from sklearn
		# Args:
		# y_true -> numpy.array(int) representing the ground truth sentiment
		# y_pred -> numpy.array(int) representing the predicted sentiment
		# Returns:
		# prints classification report and confusion matrix
		print(classification_report(y_true=y_true, y_pred=y_pred))
		print(confusion_matrix(y_true=y_true, y_pred=y_pred))

	def load_weights(self, checkpoint_dir):
		# Method to load weights from checkpoints - often used during traing in conjuction with tensorboard to monitor
		# training performance
		try:
			path = get_data(self._path_to_checkpoints) + checkpoint_dir + '\\'
			self.load_weights(path).expect_partial()
			print("Weights Loaded")  # This should be a sent to a log file down the road
		except:
			print("No checkpoints saved!")

	def save(self, version_name):
		# Method to save model to " data\\saved_models\\version"
		# Args:
		# version_dir -> of type str() representing directory name aka: V1, V2, V3...etc
		# Returns:
		# --- prints / logs a statement of success or failure

		path_to_model = get_data(self._path_to_model) + version_name
		try:
			self.save(path_to_model)
			print("model saved successfully!")
		except:
			print("Warning: model was not saved!")

	def load_model(self, version_name):
		# Method to load model from " data\\saved_models\\version"
		# Args:
		# version_dir -> of type str() representing directory name aka: V1, V2, V3...etc
		# Returns:
		# --- prints / logs a statement of success or failure
		path_to_model = os.path.join(get_data(self._path_to_model), version_name)
		try:
			reconstructed_model = keras.models.load_model(path_to_model)
			print("model loaded successfully!")
			return reconstructed_model
		except:
			print("Warning: model was not loaded!")


#############################################################################
#
# 	              pre_processing class
#
#############################################################################

class PreProcessing:
	@staticmethod
	def pre_process_pipeline(text_lst, params):
		# Method to dump embeddings matrix into config.yaml file
		# Args:
		# text   -> of type pd.DataFrame() representing the corpus we are training our model on
		# params -> dictionary() representing parameter file config.yaml
		# Returns:
		# sequences_padded -> numpy.array() representing a padded sequence
		# word_index -> dictionary mapping words to in necessary during training phase

		# parameters:
		vocab_size = params['pre_processing']['vocab_size']
		embedding_dim = params['pre_processing']['embedding_dim']
		max_length = params['pre_processing']['max_length']
		trunc_type = params['pre_processing']['trunc_type']
		padding_type = params['pre_processing']['padding_type']
		oov_tok = params['pre_processing']['oov_tok']

		# Tokenizing process
		tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
		tokenizer.fit_on_texts(text_lst)

		# Saving word_to_index dictionary
		word_index = tokenizer.word_index

		# text to sequence & padding
		sequences = tokenizer.texts_to_sequences(text_lst)
		sequences_padded = pad_sequences(sequences,
										 maxlen=max_length,
										 padding=padding_type,
										 truncating=trunc_type)

		return [sequences_padded, word_index]

	@staticmethod
	def get_embeddings_mx(word_index):
		# Method to get embeddings matrix (transfer Learning from Spacy)
		# Args:
		# word_index -> dictionary mapping each wordto an integer
		# Returns:
		# embedding_matrix: numpy.array() representing embedding matrix

		nlp = spacy.load("en_core_web_lg")
		vocab = list(word_index.keys())
		num_tokens = len(vocab)
		embedding_dim = len(nlp('The').vector)
		embedding_matrix = np.zeros((num_tokens, embedding_dim))
		for i, word in enumerate(vocab):
			embedding_matrix[i] = nlp(word).vector
		return embedding_matrix


#############################################################################
#
# 	              Interface class
#
#############################################################################

class SentimentInterface:
	# sentiment interface class to use SentimentClassifier
	def __init__(self):
		# getting parameters:
		self.sentiment_params = get_params()
		self.sentiment_classifier = SentimentClassifier(embedding_matrix=None, params=self.sentiment_params)
		self.sentiment_classifier.Load_weights('V1')

	#
	# # utility function to load dummy data
	# def get_text():
	#     path = get_data('extracted_text.parquet.gzip')
	#     text = pd.read_parquet(path).dropna()
	#     def rand_bin_array(K, N):
	#         arr = np.zeros(N)
	#         arr[:K]  = 1
	#         np.random.shuffle(arr)
	#         return arr

	#     text["Labels"] = rand_bin_array(1000,text.shape[0])
	#     text.columns=["Text","Labels"]
	#     text = text.iloc[0:100]
	#     return text

	# # just to test
	# text = get_text()  # These 2 lines are going awya (just to test)
	# txt_lst = text.Text.values.tolist()

	input_from_trend_classifier = [{'ID': 1545},
								   {'string_indices': (0, 121),
									'text': 'Three-dimensional printing has changed the way we make everything from prosthetic limbs to aircraft parts and even homes.',
									'string_prediction': ['building', '3-d printing'], 'string_prob': [0.9, 0.5]},
								   {'string_indices': (122, 181),
									'text': 'Now it may be poised to upend the apparel industry as well.',
									'string_prediction': ['building', '3-d printing'], 'string_prob': [0.9, 0.5]}]

	def text_to_sentiment(self, input_from_trend_classifier=input_from_trend_classifier):
		# Method to run end-to-end sentiment classifier
		# Args:
		# input_from_trend_classifier -> list of dictionaries at index 0 we store article ID.
		# Returns:
		# predictions -> input_from_trend_classifier list of dictionaries + in each dictionary sentiment (-1,0,1) + associated proba

		text = [input_from_trend_classifier[idx]['text'] for idx in range(len(input_from_trend_classifier)) if idx > 0]

		# preprocessing the text list
		sequences_padded, _ = PreProcessing.pre_process_pipeline(text, self.sentiment_params)

		# predicting
		y_proba, y_pred = self.sentiment_classifier.Predict(sequences_padded[0], thresh=0.6)

		# reformating y_proba and y_pred
		y_pred = y_pred.squeeze().astype(int)
		y_proba = y_proba.squeeze().astype(float)

		output = copy.deepcopy(input_from_trend_classifier)

		for idx in range(1, len(output)):
			output[idx]['sentiment_class'] = y_pred[idx]
			output[idx]['sentiment_proba'] = y_proba[idx]

		return output
