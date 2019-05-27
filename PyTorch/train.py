import os
import sys
import time
import copy
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


from model import *

def evalute(loader,net,device):
	with torch.no_grad():
		net.eval()
		loss = 0.0
		total = 0
		acc = 0.0
		for sen,asp,lab in loader:
			sen = sen.long().to(device)
			asp = asp.long().to(device)
			lab = lab.long().to(device)

			out = net(sen,asp)
			curloss = F.cross_entropy(out,lab,reduction='sum')
			loss+=curloss.item()
			preds = torch.max(out,1)[1]
			acc+=torch.sum(preds==lab.data).item()
			total+=sen.size(0)

		return curloss/total,(acc/total*100)




def train_model(trainloader,valloader,testloader,sentencembed,aspectembed,embeddim,numclasses,device,runs):

	avg_testacc = 0.0
	numepochs = 30
	for run in range(1,runs+1):
		print("Training for run {} ".format(run))
		gatedcnn = GatedCNN(sentencembed,aspectembed,embeddim,numclasses).to(device)
		optimizer = torch.optim.Adagrad(gatedcnn.parameters(), lr=0.001)

		gatedcnn.train()
		valbest = np.Inf
		best_model_wts = copy.deepcopy(gatedcnn.state_dict())
		for epoch in range(1,numepochs+1):
			gatedcnn.train()
			for sen,asp,lab in trainloader:
				sen = sen.long().to(device)
				asp = asp.long().to(device)
				lab = lab.long().to(device)

				optimizer.zero_grad()

				output = gatedcnn(sen,asp)

				loss = F.cross_entropy(output,lab)
				loss.backward()
				optimizer.step()

			valloss,valacc = evalute(valloader,gatedcnn,device)
			if(valloss<valbest):
				valbest = valloss
				best_model_wts = copy.deepcopy(gatedcnn.state_dict())

			print("Epoch {} Val Loss {} Val Acc {} ".format(epoch,valloss,valacc))

			gatedcnn.load_state_dict(best_model_wts)

		curtestloss,curtestacc = evalute(testloader,gatedcnn,device)

		print("Run {} Test Accuracy {} ".format(run,curtestacc))
		print("---------------------------------------------------")
		avg_testacc+=curtestacc

	return avg_testacc/runs		
