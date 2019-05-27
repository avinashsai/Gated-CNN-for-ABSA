import sys
import os
import re
import ast
from ast import literal_eval

label = {'negative':0,'positive':1,'neutral':2}

def preprocess(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip()

def load_data(dataset):

	temp=open(dataset+"atsa_train.json","r",encoding="ISO-8859-1").read()
	train=literal_eval(temp)
	train_sentence=[]
	train_aspect=[]
	train_sentiment=[]
	for i in train:
		if(i['sentiment']!='conflict'):
		    train_sentence.append(preprocess(i["sentence"]))
		    train_aspect.append(preprocess(i["aspect"]))
		    train_sentiment.append(label[i["sentiment"]])



	temp=open(dataset+"atsa_test.json","r",encoding="ISO-8859-1").read()
	test=literal_eval(temp)
	test_sentence=[]
	test_aspect=[]
	test_sentiment=[]
	for i in test:
		if(i['sentiment']!='conflict'):
		    test_sentence.append(preprocess(i["sentence"]))
		    test_aspect.append(preprocess(i["aspect"]))
		    test_sentiment.append(label[i["sentiment"]])


	return train_sentence,test_sentence,train_aspect,test_aspect,train_sentiment,test_sentiment