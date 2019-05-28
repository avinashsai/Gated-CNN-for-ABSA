import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCNN(nn.Module):
    def __init__(self,sen_embed,asp_embed,embeddim,numclasses):
        super(GatedCNN, self).__init__()
        
        C = numclasses
        filters = 100
        D = embeddim
        Ks = [3,4,5]
        ka = [3]

        self.sen_embed = nn.Embedding.from_pretrained(sen_embed,freeze=True)

        self.asp_embed = nn.Embedding.from_pretrained(asp_embed,freeze=True)
        
        ### Aspect Convolution
        self.conv_asp1 = nn.Conv1d(D,filters,ka[0],padding=ka[0]-2)
        ### Sentence Convolution
        self.conv_sen1 = nn.Conv1d(D,filters,Ks[0])
        self.conv_sen2 = nn.Conv1d(D,filters,Ks[1])
        self.conv_sen3 = nn.Conv1d(D,filters,Ks[2])
        ### Sentence + Aspect Convolution
        self.conv_senasp1 = nn.Conv1d(D,filters,Ks[0])
        self.conv_senasp2 = nn.Conv1d(D,filters,Ks[1])
        self.conv_senasp3 = nn.Conv1d(D,filters,Ks[2])
        
        ### Dense on Aspect
        self.fc_aspect = nn.Linear(filters, filters)
        
        ### Activations
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
        
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(len(Ks)*filters, C)
        


    def forward(self, sent, aspect):
        sentence_embed = self.sen_embed(sent)  
        aspect_embed = self.asp_embed(aspect)
        
        sentence_embed_t = sentence_embed.transpose(1,2)
        aspect_embed_t = aspect_embed.transpose(1,2)
        out_asp = self.act1(self.conv_asp1(aspect_embed_t))
        out_asp = F.max_pool1d(out_asp,out_asp.size(2)).squeeze(2)
        
        out1_sen1 = self.act2(self.conv_sen1(sentence_embed_t))
        out1_sen2 = self.act2(self.conv_sen2(sentence_embed_t))
        out1_sen3 = self.act2(self.conv_sen3(sentence_embed_t))
        
        asp_ful = self.fc_aspect(out_asp).unsqueeze(2)
        out2_sen1 = self.act1((self.conv_senasp1(sentence_embed_t))+asp_ful)
        out2_sen2 = self.act1((self.conv_senasp2(sentence_embed_t))+asp_ful)
        out2_sen3 = self.act1((self.conv_senasp3(sentence_embed_t))+asp_ful)

        out_comb1 = out1_sen1 * out2_sen1
        out_comb2 = out1_sen2 * out2_sen2
        out_comb3 = out1_sen3 * out2_sen3
        
        out_comb1 = F.max_pool1d(out_comb1,out_comb1.size(2)).squeeze(2)
        out_comb2 = F.max_pool1d(out_comb2,out_comb2.size(2)).squeeze(2)
        out_comb3 = F.max_pool1d(out_comb3,out_comb3.size(2)).squeeze(2)
        
        out = torch.cat([out_comb1,out_comb2,out_comb3],dim=1)
        out = self.dropout(out)  
        out = self.fc1(out)
        return out