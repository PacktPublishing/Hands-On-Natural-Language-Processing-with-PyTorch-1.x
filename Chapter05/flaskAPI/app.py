import flask
from flask import Flask, jsonify, request
import json

import pandas as pd
from string import punctuation
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim

app = Flask(__name__)
@app.route('/predict', methods=['GET'])

def predict():

	class SentimentLSTM(nn.Module):
    
	    def __init__(self, n_vocab = 5401, n_embed = 50, n_hidden = 100, n_output = 1, n_layers = 2, drop_p = 0.8):
	        super().__init__()
	        
	        self.n_vocab = n_vocab     
	        self.n_layers = n_layers   
	        self.n_hidden = n_hidden   
	        
	        self.embedding = nn.Embedding(n_vocab, n_embed)
	        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first = True, dropout = drop_p)
	        self.dropout = nn.Dropout(drop_p)
	        self.fc = nn.Linear(n_hidden, n_output)
	        self.sigmoid = nn.Sigmoid()
	        
	        
	    def forward (self, input_words):                              
	        embedded_words = self.embedding(input_words)   
	        lstm_out, h = self.lstm(embedded_words)     
	        lstm_out = self.dropout(lstm_out)
	        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)
	        fc_out = self.fc(lstm_out)                      
	        sigmoid_out = self.sigmoid(fc_out)              
	        sigmoid_out = sigmoid_out.view(batch_size, -1)  
	        
	        
	        sigmoid_last = sigmoid_out[:, -1] 
	        
	        return sigmoid_last, h
	    
	    
	    def init_hidden (self, batch_size):
	        device = "cpu"
	        weights = next(self.parameters()).data
	        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
	             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
	        
	        return h

	def preprocess_review(review):
		with open('models/word_to_int_dict.json') as handle:
			word_to_int_dict = json.load(handle)

		review = review.translate(str.maketrans('', '', punctuation)).lower().rstrip()
	    
		tokenized = word_tokenize(review)

		if len(tokenized) >= 50:
			review = tokenized[:50]
		else:
			review= ['0']*(50-len(tokenized)) + tokenized

		final = []

		for token in review:
			try:
				final.append(word_to_int_dict[token])

			except:
				final.append(word_to_int_dict[''])

		return final  


	request_json = request.get_json()
	i = request_json['input']

	batch_size = 1

	model = SentimentLSTM(5401, 50, 100, 1, 2)

	model.load_state_dict(torch.load("models/model_nlp.pkl"))
	model.eval()
	words = np.array([preprocess_review(review=i)])
	padded_words = torch.from_numpy(words)
	pred_loader = DataLoader(padded_words, batch_size = 1, shuffle = True)
	for x in pred_loader:
		output = model(x)[0].item()

	response = json.dumps({'response': output})
	return response, 200
