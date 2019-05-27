import os
import re
import sys
import collections
from collections import Counter
from sklearn.model_selection import *
import numpy as np 
import torch
import torch.utils.data
from torch.utils.data import Dataset

senlen = 83
asplen = 9
batchsize = 32

def get_vocab(data):
	words = []
	for sentence in data:
		words+=sentence.split()

	counts = Counter(words).most_common()

	vocabulary = {}
	vocabulary['PAD'] = 0
	index = 1
	for word,_ in counts:
		vocabulary[word] = index
		index+=1

	return vocabulary

def convert_indices(sentence,vocab,maxlen):
	corpusind = [vocab[word] for word in sentence.split() if word in vocab]
	padind = [0]*maxlen
	curlen = len(corpusind)
	if(maxlen-curlen<0):
		padind = corpusind[:maxlen]
	else:
		padind[maxlen-curlen:] = corpusind

	return torch.from_numpy(np.asarray(padind,dtype='int32'))


def get_indices(data,vocab,maxlen):
	indices = torch.zeros(len(data),maxlen)
	for i in range(len(data)):
		indices[i] = convert_indices(data[i],vocab,maxlen)

	return indices

def generate_batches(trainsen,Xtestsen,trainasp,Xtestasp,trainl,ytest):
	Xtrainsen,Xvalsen,Xtrainasp,Xvalasp,ytrain,yval = train_test_split(trainsen,trainasp,trainl,
		test_size=0.1,random_state=42)

	senvocab = get_vocab(Xtrainsen)
	aspvocab = get_vocab(Xtrainasp)

	trainsenind = get_indices(Xtrainsen,senvocab,senlen)
	trainaspind = get_indices(Xtrainasp,aspvocab,asplen)


	valsenind = get_indices(Xvalsen,senvocab,senlen)
	valaspind = get_indices(Xvalasp,aspvocab,asplen)

	testsenind = get_indices(Xtestsen,senvocab,senlen)
	testaspind = get_indices(Xtestasp,aspvocab,asplen)

	ytrain = torch.from_numpy(np.asarray(ytrain,'int32'))
	yval = torch.from_numpy(np.asarray(yval,'int32'))
	ytest = torch.from_numpy(np.asarray(ytest,'int32'))


	trainarray = torch.utils.data.TensorDataset(trainsenind,trainaspind,ytrain)
	trainloader = torch.utils.data.DataLoader(trainarray,batchsize)
	
	valarray = torch.utils.data.TensorDataset(valsenind,valaspind,yval)
	valloader = torch.utils.data.DataLoader(valarray,batchsize)
	
	testarray = torch.utils.data.TensorDataset(testsenind,testaspind,ytest)
	testloader = torch.utils.data.DataLoader(testarray,batchsize)
	
	return trainloader,valloader,testloader,senvocab,aspvocab