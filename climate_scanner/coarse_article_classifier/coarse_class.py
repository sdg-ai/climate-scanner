import json
import pandas as pd
import string
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import copy
import tqdm

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)
    
df = pd.read_csv(get_data('non_climate_data.csv'))

train_df, test_df = train_test_split(df, test_size=0.20, random_state = 42)
train_df, val_df = train_test_split(train_df, test_size=0.02, random_state = 42)

# Import the BertTokenizer from the library
from transformers import BertTokenizer

# Load a pre-trained BERT Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from torch.utils.data import Dataset, DataLoader

class SST2BertDataset(Dataset):
    
    def __init__(self, sentences, labels, seq_len, bert_variant = "bert-base-uncased"):
        """
        Constructor for the `SST2BertDataset` class. Stores the `sentences` and `labels` which can then be used by
        other methods. Also initializes the tokenizer
        
        Inputs:
            - sentences (list) : A list of movie reviews
            - labels (list): A list of sentiment labels corresponding to each review
            - seq_len (int): Length of the sequence to use.
                             If number of tokens are lower than `seq_len` add padding otherwise truncate
        """
        self.sentences = sentences
        self.labels = labels
        self.seq_len = seq_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_variant)
        
    def __len__(self):
        """
        Returns the length of the dataset i.e. the number of reviews present in the dataset
        """
        length = len(self.sentences)
        return length
    
    def __getitem__(self, idx):
        """
        Returns the training example corresponding to review present at the `idx` position in the dataset
        
        Inputs:
            - idx (int): Index corresponding to the review,label to be returned
            
        Returns:
            - input_ids (torch.tensor): Indices of the tokens in the sentence at `idx` position.
                                        Shape of the tensor should be (`seq_len`,)
            - mask (torch.tensor): Attention mask indicating which tokens are padded.
                                   Shape of the tensor should be (`seq_len`,)
            - label (int): Sentiment label for the corresponding sentence
        
        Hint: To get the output from the tokenizer in the form of torch tensors set return_tensors="pt" when calling self.tokenizer 
        """
        
        tokenizer_output = self.tokenizer(self.sentences[idx], max_length=self.seq_len, padding="max_length", truncation = True, return_tensors="pt")
        input_ids = tokenizer_output["input_ids"]
        mask = tokenizer_output["attention_mask"]
        label = self.labels[idx]
        
        return input_ids.squeeze(0), mask.squeeze(0), label

seq_len = 128
batch_size = 16

train_sentences, train_labels = train_df["sentence"].values, train_df["label"].values
val_sentences, val_labels = val_df["sentence"].values, val_df["label"].values
test_sentences, test_labels = test_df["sentence"].values, test_df["label"].values

train_dataset = SST2BertDataset(train_sentences, train_labels, seq_len=seq_len)
val_dataset = SST2BertDataset(val_sentences, val_labels, seq_len=seq_len)
test_dataset = SST2BertDataset(test_sentences, test_labels, seq_len=seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

from transformers import BertModel

bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model

class BertClassifierModel(nn.Module):
    
    def __init__(self, d_hidden = 768, bert_variant = "bert-base-uncased"):
        """
        Define the architecture of Bert-Based classifier.
        You will mainly need to define 3 components, first a BERT layer
        using `BertModel` from transformers library,
        a linear layer to map the representation from Bert to the output,
        and a sigmoid layer to map the score to a proability
        
        Inputs:
            - d_hidden (int): Size of the hidden representations of bert
            - bert_variant (str): BERT variant to use
        """
        super(BertClassifierModel, self).__init__()
        
        self.bert_layer = BertModel.from_pretrained(bert_variant)
        self.output_layer = nn.Linear(d_hidden, 1)
        self.sigmoid_layer = nn.Sigmoid()
        
    def forward(self, input_ids, attn_mask):
        """
        Forward Passes the inputs through the network and obtains the prediction
        
        Inputs:
            - input_ids (torch.tensor): A torch tensor of shape [batch_size, seq_len]
                                        representing the sequence of token ids
            - attn_mask (torch.tensor): A torch tensor of shape [batch_size, seq_len]
                                        representing the attention mask such that padded tokens are 0 and rest 1
                                        
        Returns:
          - output (torch.tensor): A torch tensor of shape [batch_size,] obtained after passing the input to the network
                                        
        
        Hint: Recall which of the outputs from BertModel is appropriate for the sentence classification task.
        """

        # pooler_output is an aggregate representation of the entire sentence and can be thought of as a sentence embedding. 
        # It is obtained by passing the representation of the [CLS] token through a linear layer. 
        # This can be useful for sentence-level tasks like sentiment analysis etc.

        output = self.bert_layer(input_ids, attention_mask = attn_mask).pooler_output
        output = self.sigmoid_layer(self.output_layer(output))
        
        return output.squeeze(-1) 

# HELPER FUNCTIONS

def get_accuracy(pred_labels, act_labels):
    """
    Calculates the accuracy value by comparing predicted labels with actual labels

    Inputs:
    - pred_labels (numpy.ndarray) : A numpy 1d array containing predicted labels. 
    - act_labels (numpy.ndarray): A numpy 1d array containing actual labels (of same size as pred_labels). 

    Returns:
    - accuracy (float): Number of correct predictions / Total number of predictions

    """
    accuracy = sum(pred_labels == act_labels)/len(pred_labels)
    return accuracy

def convert_probs_to_labels(probs, threshold = 0.5):
    """
    Convert the probabilities to labels by using the specified threshold

    Inputs:
    - probs (numpy.ndarray): A numpy 1d array containing the probabilities predicted by the classifier model
    - threshold (float): A threshold value beyond which we assign a positive label i.e 1 and 0 below it

    Returns:
    - labels (numpy.ndarray): Labels obtained after thresholding

    """
    labels = np.where(probs > threshold, 1, 0)
    return labels

def evaluate_model_metrics(model, test_dataloader, threshold = 0.5, device = "cpu"):
    """
    Evaluates `model` on test dataset

    Inputs:
        - model (BertClassifierModel): Logistic Regression model to be evaluated
        - test_dataloader (torch.utils.DataLoader): A dataloader defined over the test dataset
        - threshold (float): Probability Threshold above which we consider label as 1 and 0 below

    Returns:
        - precision (float): Precision Score over the test dataset 
        - recall (float): Recall Score over the test dataset 
        - F1 score (float): F1 Score (HM of precision and recall) over the test dataset         
        - accuracy (float): Average accuracy over the test dataset     
    """
    
    model.eval()
    model = model.to(device)
    accuracy = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for test_batch in test_dataloader:
            features, mask, labels = test_batch
            features = features.float().to(device).long()
            mask = mask.float().to(device)
            labels = labels.float().to(device)
            pred_probs = model(features, mask)
            pred_probs = pred_probs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            pred_labels = convert_probs_to_labels(pred_probs, threshold)
            batch_accuracy = get_accuracy(convert_probs_to_labels(pred_probs, threshold), labels)
            accuracy += batch_accuracy
            confusion_vector = torch.from_numpy(pred_labels / labels)
            true_positives += torch.sum(confusion_vector == 1).item()
            false_positives += torch.sum(confusion_vector == float('inf')).item()
            true_negatives += torch.sum(torch.isnan(confusion_vector)).item()
            false_negatives += torch.sum(confusion_vector == 0).item()
        accuracy = accuracy / len(test_dataloader)
        recall = true_positives/(true_positives+false_negatives)
        precision = true_positives/(true_positives+false_positives)
        f1_score = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1_score, accuracy

def evaluate(model, test_dataloader, threshold = 0.5, device = "cpu"):
    """
    Evaluates `model` on test dataset

    Inputs:
        - model (BertClassifierModel): Logistic Regression model to be evaluated
        - test_dataloader (torch.utils.DataLoader): A dataloader defined over the test dataset
        - threshold (float): Probability Threshold above which we consider label as 1 and 0 below

    Returns:
        - accuracy (float): Average accuracy over the test dataset 
    """
    
    model.eval()
    model = model.to(device)
    accuracy = 0
    
    with torch.no_grad():
        for test_batch in test_dataloader:
            features, mask, labels = test_batch
            features = features.float().to(device).long()
            mask = mask.float().to(device)
            labels = labels.float().to(device)
            pred_probs = model(features, mask)
            pred_probs = pred_probs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            batch_accuracy = get_accuracy(convert_probs_to_labels(pred_probs, threshold), labels)
            accuracy += batch_accuracy
        accuracy = accuracy / len(test_dataloader)
    
    return accuracy

def train(model, train_dataloader, val_dataloader,
          lr = 1e-5, num_epochs = 3,
          device = "cpu"):
    """
    Runs the training loop. Define the loss function as BCELoss like the last tine
    and optimizer as Adam and traine for `num_epochs` epochs.

    Inputs:
        - model (BertClassifierModel): BERT based classifer model to be trained
        - train_dataloader (torch.utils.DataLoader): A dataloader defined over the training dataset
        - val_dataloader (torch.utils.DataLoader): A dataloader defined over the validation dataset
        - lr (float): The learning rate for the optimizer
        - num_epochs (int): Number of epochs to train the model for.
        - device (str): Device to train the model on. Can be either 'cuda' (for using gpu) or 'cpu'

    Returns:
        - best_model (BertClassifierModel): model corresponding to the highest validation accuracy (checked at the end of each epoch)
        - best_val_accuracy (float): Validation accuracy corresponding to the best epoch
    """
    epoch_loss = 0
    model = model.to(device)
    
    best_val_accuracy = float("-inf")
    best_model = None
    
    # 1. Define Loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr)
    
    for epoch in range(num_epochs):
        model.train() # Since we are evaluating model at the end of every epoch, it is important to bring it back to train mode
        epoch_loss = 0
        
        # 2. Write Training Loop (store the loss for each batch in epoch_loss like done in previous assignments)
        for train_batch in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()
            features, mask, labels = train_batch
            features = features.float().to(device).long()
            mask = mask.float().to(device)
            labels = labels.float().to(device)
            preds = model(features, mask)
            loss = loss_fn(preds, labels)
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / len(train_dataloader)
        
        # 3. Evaluate on validation data by calling `evaluate` and store the validation accuracy in `val_accurracy`
        val_accuracy = evaluate(model, val_dataloader, device = device)
        
        # Model selection
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model) # Create a copy of model
        
        print(f"Epoch {epoch} completed | Average Training Loss: {epoch_loss} | Validation Accuracy: {val_accuracy}")
 
    return best_model, best_val_accuracy

torch.manual_seed(42)

model = BertClassifierModel()
best_model, best_val_acc = train(model, train_loader, val_loader, num_epochs = 3, device = "cuda")

test_accuracy = evaluate(best_model, test_loader, threshold = 0.5, device = "cuda")
print(test_accuracy)

def predict_text(text, model, tokenizer, threshold = 0.5, device = "cpu"):
    """
    Predicts the sentiment label for a piece of text using the BERT classifier model
    
    Inputs:
        - text (str): The sentence/document whose sentiment is to be predicted
        - model (BertClassifierModel): Fine-tuned BERT based classifer model
        - tokenizer (BertTokenizer): Pre-trained BERT tokenizer
        - threshold (float): Probability Threshold above which we consider label as 1 and 0 below
    Returns:
        - pred_label (float): Predicted sentiment of the document
    """
    
    model = model.to(device)
    model.eval()
    tokenizer_output = tokenizer(text, return_tensors="pt")
    input_ids, attn_mask = tokenizer_output["input_ids"], tokenizer_output["attention_mask"]
    prob = model(input_ids, attn_mask)
    pred_label = convert_probs_to_labels(prob, threshold)
    
    return pred_label

def test(text):
    return predict_text(text, best_model, bert_tokenizer)
