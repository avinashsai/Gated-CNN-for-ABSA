import os
import sys
import re
import argparse
import random
import numpy as np 
from sklearn.model_selection import *
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F 

from loader import *
from convert import * 
from w2v import *
from model import *
from train import *

np.random.seed(1332)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

datapath = '../atsa-'
embedpath = '../glove.840B.300d.txt'

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-da','--dataset',type=str,help='dataset',default='restaurant')
	parser.add_argument('-ru','--runs',type=int,help='number of runs',default=5)

	args = parser.parse_args()
	dataset = args.dataset
	runs = args.runs

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	traincorpus,testcorpus,train_aspect,test_aspect,trainlabels,testlabels = load_data(datapath+dataset+"/")

	numclasses = max(trainlabels)+1

	trainloader,valloader,testloader,senvocab,aspvocab = generate_batches(traincorpus,testcorpus,train_aspect,test_aspect,trainlabels,testlabels)

	embedding_index,embed_dim = load_embed(embedpath)

	sentenceembed,aspectembed = load_embeddings(embedding_index,embed_dim,senvocab,aspvocab)

	gated_cnn = GatedCNN(sentenceembed,aspectembed,embed_dim,numclasses).to(device)

	test_acc = train_model(trainloader,valloader,testloader,sentenceembed,aspectembed,embed_dim,numclasses,device,runs)

	print("Average Test Accuracy {} ".format(test_acc))