import os
import re
import sys
import numpy as np 
import torch
from torch.distributions import uniform

def load_embed(embed_path):

	embedding_index = {}
	with open(embed_path,'r',encoding='utf-8') as f:
		for line in f.readlines():
			lexicons = line.split(' ')
			word = lexicons[0]
			embedding = torch.from_numpy(np.asarray(lexicons[1:],dtype='float32'))
			embedding_index[word] = embedding
	embed_dim = int(embedding.size()[0])

	return embedding_index,embed_dim


def load_embeddings(embedding_index,embed_dim,senvocab,aspvocab):

	sentence_embed = torch.zeros(len(senvocab),embed_dim)
	i = 0
	for word in senvocab.keys():
		if(word not in embedding_index):
			if(word!='PAD'):
				sentence_embed[i,:] = uniform.Uniform(-0.25,0.25).sample(torch.Size([embed_dim]))
		else:
			sentence_embed[i,:] = embedding_index[word]
		i+=1

	
	aspect_embed = torch.zeros(len(aspvocab),embed_dim)
	i = 0
	for word in aspvocab.keys():
		if(word not in embedding_index):
			if(word!='PAD'):
				aspect_embed[i,:] = uniform.Uniform(-0.25,0.25).sample(torch.Size([embed_dim]))
		else:
			aspect_embed[i,:] = embedding_index[word]
		i+=1

	return sentence_embed,aspect_embed