import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCNN(nn.Module):
	def __init__(self,senembed,aspembed,embeddim,numclasses):
		super(GatedCNN,self).__init__()
		self.senembed = senembed
		self.aspembed = aspembed
		self.embeddim = embeddim
		self.numfilters = 100
		self.filters1 = [3,4,5]
		self.filters2 = [3]
		self.lineardim = 100
		self.numclasses = numclasses

		self.sen_embed = nn.Embedding.from_pretrained(self.senembed,freeze=True)
		self.asp_embed = nn.Embedding.from_pretrained(self.aspembed,freeze=False)

		self.convs1 = nn.ModuleList([nn.Conv1d(self.embeddim, self.numfilters, K) for K in self.filters1])
		self.convs2 = nn.ModuleList([nn.Conv1d(self.embeddim, self.numfilters, K) for K in self.filters1])
		self.convs3 = nn.ModuleList([nn.Conv1d(self.embeddim, self.numfilters, K, padding=K-2) for K in self.filters2])

		self.drop = nn.Dropout(0.2)
		self.linear = nn.Linear(self.lineardim,self.numfilters)

		self.final = nn.Linear(len(self.filters1)*self.numfilters,self.numclasses)


	def forward(self,xsen,xasp):
		senembed_out = self.sen_embed(xsen)  
		aspembed_out = self.asp_embed(xasp)  
		aa = [F.relu(conv(aspembed_out.transpose(1, 2))) for conv in self.convs3]  
		aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
		aspembed_out = torch.cat(aa, 1)

		x = [torch.tanh(conv(senembed_out.transpose(1, 2))) for conv in self.convs1]  
		y = [F.relu(conv(senembed_out.transpose(1, 2)) + self.linear(aspembed_out).unsqueeze(2)) for conv in self.convs2]
		x = [i*j for i, j in zip(x, y)]

		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  

		x = torch.cat(x, 1)
		x = self.drop(x)  
		out = self.final(x)  
		return out