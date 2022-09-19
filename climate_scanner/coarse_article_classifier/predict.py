import numpy as np
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel

_ROOT = os.path.abspath(os.path.dirname(__file__))

def load_weights(model, path):
    path1 = os.path.join(_ROOT, path)
    model.load_state_dict(torch.load(path1))
    return model

def convert_probs_to_labels(probs, threshold = 0.5):
    labels = np.where(probs > threshold, 1, 0)
    return labels

class BertClassifierModel(nn.Module):
    def __init__(self, d_hidden = 768, bert_variant = "bert-base-uncased"):
        super(BertClassifierModel, self).__init__()
        self.bert_layer = BertModel.from_pretrained(bert_variant)
        self.output_layer = nn.Linear(d_hidden, 1)
        self.sigmoid_layer = nn.Sigmoid()
        
    def forward(self, input_ids, attn_mask):
        output = self.bert_layer(input_ids, attention_mask = attn_mask).pooler_output
        output = self.sigmoid_layer(self.output_layer(output))
        return output.squeeze(-1)

def predict_text(text, model, tokenizer, threshold = 0.5, device = "cpu"):
    model = model.to(device)
    model.eval()
    tokenizer_output = tokenizer(text, return_tensors="pt")
    input_ids, attn_mask = tokenizer_output["input_ids"], tokenizer_output["attention_mask"]
    prob = model(input_ids, attn_mask)
    pred_label = convert_probs_to_labels(prob, threshold)
    return pred_label

model = BertClassifierModel()
model = load_weights(model, "checkpoint.pth")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
predict_text("I love you", model, bert_tokenizer)