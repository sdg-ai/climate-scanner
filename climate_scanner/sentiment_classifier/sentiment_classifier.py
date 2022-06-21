# -*- coding: utf-8 -*-

import os
import copy
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
# optimizaer 
from transformers import AdamW


from torch.utils.data import TensorDataset, DataLoader, RandomSampler,SequentialSampler
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

class BERT_arch(nn.Module):
	def __init__(self,device):
		super(BERT_arch,self).__init__()

		# load bert and tokanizer
		self.bert = AutoModel.from_pretrained('bert-base-uncased')
		self.device = device
		# freeze all bert params
		for param in self.bert.parameters():
			param.requires_grad=False

		# dropout layer
		self.dropout = nn.Dropout(0.1)

		# relu activation
		self.relu = nn.ReLU()

		# Dense Layer 1 
		self.fc1 = nn.Linear(768,500)

		# Dense Layer 2
		self.fc2 = nn.Linear(500,300)

		# Dense Layer 3
		self.fc3 = nn.Linear(300,156)

		#Dense Layer 4 
		self.fc4 = nn.Linear(156,2)

		# Soft max
		self.softmax = nn.Softmax(dim=1)

		#loss function:
		self.cross_entropy = nn.CrossEntropyLoss()

		

	def forward(self,sent_id,mask):
		# pass inputs to the model
		_,cls_hs = self.bert(sent_id,attention_mask=mask,return_dict=False)
		x = self.fc1(cls_hs)

		x = self.relu(x)

		x = self.dropout(x)

		x = self.fc2(x)

		x = self.fc3(x)

		x = self.fc4(x)

		x = self.softmax(x)

		return x
  
	def fit(self,train_dataloader):
		self.train()

		total_loss, total_accuracy = 0,0

		# empty list to save model predictions
		total_preds=[]

		# iterate over batches
		for step,batch in enumerate(train_dataloader):
			# progress update after every 50 batches
			if step%50==0 and step!=0:
				print(' Batch {:>5,}  of {:>5,}.'.format(step,len(train_dataloader)))

			# push batch to GPU
			batch = [r.to(self.device) for r in batch]

			sent_id,mask,labels=batch

			# clear previously calculated gradients
			self.zero_grad()

			# get model prediction for current batch
			preds = self(sent_id,mask)

			# compute the loss
			loss = self.cross_entropy(preds,labels)

			# add on the total loss
			total_loss = total_loss + loss.item()

			# backward pass to calculate gradients
			loss.backward()

			# clip gradients to 1 to avoid exploding scenarios
			# torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

			# update parameters
			self.optimizer.step()

			# model predictions are stored in GPU so need to push them back to CPU
			preds = preds.detach().cpu().numpy()

			# append model prediction
			total_preds.append(preds)
		# compute training loss of the epoch
		avg_loss = total_loss/len(train_dataloader)

		# predictions are in the form of (no. of batches. size of batches. no of classes)
		#reshape the predictions to (number of samples, no of classes)

		total_preds = np.concatenate(total_preds,axis=0)

		return avg_loss,total_preds


	def evaluate(self,train_dataloader,val_dataloader):
		print('\nEvaluating...')

		# desactivate droput layer
		self.eval()

		total_loss,total_accuracy = 0,0

		#empty list for predictions
		total_preds=[]

		for step,batch in enumerate(val_dataloader):
			#progress update
			if step%50==0 and step!=0:
				#get elapsed time
				# elapsed = format_time(time.time()-t0)

				# report progress
				print(' Batch {:>5,}  of {:>5,}.'.format(step,len(train_dataloader)))
			
			# push batch to GPU
			batch = [r.to(self.device) for r in batch]

			sent_id,mask,labels=batch

			with torch.no_grad():
				preds = self(sent_id,mask)

			#compute validation loss
			loss = self.cross_entropy(preds,labels)

			total_loss = total_loss + loss.item()

			preds = preds.detach().cpu().numpy()

			total_preds.append(preds)
		avg_loss = total_loss / len(val_dataloader)

		#reshaping
		total_preds = np.concatenate(total_preds,axis=0)

		return avg_loss,total_preds


	def predict(self,X_seq,X_mask,thresh=None):
		# Method to predict sentiment around text
		# Args: 
		# 	X      -> dictionary with : input_ids and attention_masks lists
		# 	thresh -> tuple of two floats (lowBound,upBound) temporarely used to determine Neutral Sentiment
		# returns:
		# 	y_proba -> numpy.array() representing vector of probabilities for each class
		# 	y_pred  -> numpy.array() representing vector of predicted classes

		
		scores_tensors = self(X_seq,X_mask)

		scores = scores_tensors.tolist()

		preds=[]
		for elem in scores:
			if elem[1]>thresh[0] and elem[1]<thresh[1]:
				preds.append(("Neutral",elem[1]))
			elif elem[1]>elem[0]:
				preds.append(('Positive',elem[1]))
			else:
				preds.append(('Negative',elem[1]))
		return preds
#############################################################################
#
# 	              Training class
#
#############################################################################

class Training:
	@staticmethod
	def train(model,device,train_dataloader,val_dataloader,epochs):
		'''
		Method to train Bert Model 
		inputs: 
			model  -> pytorch model 
			epochs -> int representing the number of epochs
			device -> torch device (cpu vs cuda)
		'''
		# set initoial loss
		best_valid_loss = float('inf')

		# empty lists to store training and validation losses
		train_losses=[]
		valid_losses=[]

		for epoch in range(epochs):
			print('\nEpoch {:} / {:}'.format(epoch+1,epochs))

			#train model
			train_loss,_=model.fit(train_dataloader)

			# evaluate model
			valid_loss,_ = model.evaluate(train_dataloader,val_dataloader)

			# save best model loss
			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				torch.save(model.state_dict(),'/content/drive/MyDrive/ai-for-good/saved_weights3.pt')

			#append training and validation losses
			train_losses.append(train_loss)
			valid_losses.append(valid_loss)

			print(f'\nTraining Loss: {train_loss:.3f}')
			print(f'\nValidation Loss: {valid_loss:.3f}')


#############################################################################
#
# 	              pre_processing class
#
#############################################################################
	
class PreProcessing:
	def __init__(self,device):
		self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		self.device=device
	
	def get_data_loader(self,txt_list,labels,batch_size,max_token_length):
		'''
		Method that creates dataloaders
		
		'''
		# tokenize
		tokens = self.tokenizer.batch_encode_plus(txt_list.tolist(),
										  max_length =max_token_length,
										  pad_to_max_length=True,
										  add_special_tokens=True,
										  truncation=True,
										  return_token_type_ids=False)

		
		# Cover Integer sequences to Tensors
		seq = torch.tensor(tokens['input_ids'])
		mask = torch.tensor(tokens['attention_mask'])
		y = torch.tensor(labels.tolist())

		

		# combine Tensors
		data = TensorDataset(seq,mask,y)

		# sampler for sampling during training process
		sampler = RandomSampler(data)

		# data loader for train set
		dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)

		return dataloader




	def pre_process_pipeline(self,text_lst):
		# Method to dump embeddings matrix into config.yaml file
		# Args:
		# text   -> of type list of strings
		# Returns:
		# X_seq	 -> tensor of padded tokens
		# X_mask -> tensor of attention masks 
		
		# Tokenizing process
		tokens = self.tokenizer.batch_encode_plus(text_lst,
										  max_length = 100,
										  pad_to_max_length=True,
										  add_special_tokens=True,
										  truncation=True,
										  return_token_type_ids=False)

		X_seq = torch.tensor(tokens['input_ids']).to(self.device)
		X_mask = torch.tensor(tokens['attention_mask']).to(self.device)

		return X_seq,X_mask


#############################################################################
#
# 	              Interface class
#
#############################################################################

class SentimentInterface:
	# sentiment interface class to use SentimentClassifier
	def __init__(self):
		# path to model 
		path_to_model = get_data(os.path.join('model', 'model_v1.pt'))
		device = torch.device('cpu')
		# model
		self.model = BERT_arch(device)
		self.model.to(device)
		# loading weights
		self.model.load_state_dict(torch.load(path_to_model,map_location=device))
		
		# preprocessing class
		self.preProcess = PreProcessing(device)
		
		# attaching model to device
		self.model.to(device)
		

	def input_to_sentiment(self, input_from_trend_classifier):
		# Method to run end-to-end sentiment classifier
		# Args:
		# input_from_trend_classifier -> list of dictionaries at index 0 we store article ID.
		# Returns:
		# predictions -> input_from_trend_classifier list of dictionaries + in each dictionary sentiment (-1,0,1) + associated proba

		text_lst = [input_from_trend_classifier[idx]['text'] for idx in range(len(input_from_trend_classifier)) if idx > 0]

		# preprocessing the text list
		X_seq,X_mask = self.preProcess.pre_process_pipeline(text_lst)

		# predicting
		y_hat = self.model.predict(X_seq,X_mask,thresh=(0.45,0.55))

		# reformating y_proba and y_pred
		# commenting for now
		# y_pred = y_pred.squeeze().astype(int)
		# y_proba = y_proba.squeeze().astype(float)

		output = copy.deepcopy(input_from_trend_classifier)

		for idx in range(1,len(output)):
			# output[idx]['sentiment_class'] = y_pred[idx]
			# output[idx]['sentiment_proba'] = y_proba[idx]
			output[idx]['sentiment_class'] = y_hat[idx-1][0]
			output[idx]['sentiment_proba'] = y_hat[idx-1][1]

		
		
		return output


	def text_to_sentiment(self, text_lst):
		if isinstance(text_lst, str):
			text_lst = [text_lst]

		# preprocessing the text list
		X_seq, X_mask = self.preProcess.pre_process_pipeline(text_lst)

		# predicting
		y_hat = self.model.predict(X_seq, X_mask, thresh=(0.45, 0.55))

		# reformating y_proba and y_pred
		# commenting for now
		# y_pred = y_pred.squeeze().astype(int)
		# y_proba = y_proba.squeeze().astype(float)

		# output = []
        #
		# for idx in range(0, len(text_lst)):
		# 	# output[idx]['sentiment_class'] = y_pred[idx]
		# 	# output[idx]['sentiment_proba'] = y_proba[idx]
		# 	output[idx] = {}
		# 	output[idx]['sentiment_class'] = y_hat[idx - 1][0]
		# 	output[idx]['sentiment_proba'] = y_hat[idx - 1][1]

		return y_hat

if __name__ == '__main__':	

	x = SentimentInterface()
	input_from_trend_classifier = [{'ID': 1545},
									{'string_indices': (0, 121),
										'text': 'Three-dimensional printing has changed the way we make everything from prosthetic limbs to aircraft parts and even homes.',
										'string_prediction': ['building', '3-d printing'], 'string_prob': [0.9, 0.5]},
									{'string_indices': (122, 181),
										'text': 'This is not a great start for the industry!.',
										'string_prediction': ['building', '3-d printing'], 'string_prob': [0.9, 0.5]},
									{'string_indices': (123, 154),
										'text': 'windmills',
										'string_prediction': ['building', '3-d printing'], 'string_prob': [0.9, 0.5]}]
	#Now it may be poised to upend the apparel industry as well.
	output = x.input_to_sentiment(input_from_trend_classifier=input_from_trend_classifier)
	print("TEST:\n\n")
	for i in range(1,len(output)):
		print('TEXT: '+output[i]['text']+ '\nsentiment_class:  '+output[i]['sentiment_class'] + '\nProba: ' + str(round(output[i]['sentiment_proba'],6))+'\n\n')
