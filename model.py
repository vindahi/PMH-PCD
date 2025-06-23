import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F
import math
import numpy as np



class Fusion(nn.Module):
    def __init__(self, fusion_dim, nbit):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        self.hash = nn.Sequential(
            nn.Linear(fusion_dim, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh()
        )

    def forward(self, x, y):
        fused_feat = self.fusion(torch.cat([x, y], dim=-1))
        hash_code = self.hash(fused_feat)
        return hash_code


class PCH(nn.Module):
    def __init__(self, args):
        super(PCH, self).__init__()
        self.image_dim = args.image_dim
        self.text_dim = args.text_dim

        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        self.nbit = int(args.nbit)
        self.classes = args.classes
        self.batch_size = args.batch_size
        
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]

        # self.dropout = args.dropout
        self.fusionnn = Fusion(fusion_dim=self.common_dim, nbit=self.nbit)

        # self.imageMLP = MLP(hidden_dim=self.img_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)

        # self.textMLP = MLP(hidden_dim=self.txt_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)
        self.imageMLP = nn.Linear(self.image_dim, self.common_dim)
        self.textMLP = nn.Linear(self.text_dim, self.common_dim)

        self.ifeat_gate = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim*2),
            nn.ReLU(),
            nn.Linear(self.common_dim*2, self.common_dim), 
            nn.Sigmoid())
        self.tfeat_gate = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim*2),
            nn.ReLU(),
            nn.Linear(self.common_dim*2, self.common_dim),
            nn.Sigmoid())
        self.pro = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim*2),
            nn.ReLU(),
            nn.Linear(self.common_dim*2, self.nbit), 
            nn.BatchNorm1d(self.nbit),
            nn.Sigmoid())
        self.activation = nn.ReLU()
        self.neck = nn.Sequential(
            nn.Linear(self.common_dim,self.common_dim*4),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.common_dim*4,self.common_dim)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.nbit, self.common_dim),
            nn.ReLU()
        )

        self.hash_output = nn.Sequential(
            nn.Linear(self.common_dim, self.nbit),
            nn.Tanh())
        self.classify = nn.Linear(self.nbit, self.classes)

    def forward(self, image, text, tgt=None):
        self.batch_size = len(image)
        imageH = self.imageMLP(image)#nbit length
        textH = self.textMLP(text)

        pimage = self.pro(imageH) 
        ptext = self.pro(textH) 
        fused_fine = self.fusionnn(imageH, textH)

        cfeat_concat = self.fusion_layer(fused_fine)
        # cfeat_concat = self.activation(cfeat_concat)
        # nec_vec = self.neck(cfeat_concat)     
        code = self.hash_output(cfeat_concat)   
        return pimage, ptext, code, self.classify(code)


class classone(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter((torch.randn(args.classes, args.nbit) / 8))
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mrg = 1.0

    def forward(self, feature_1, predict_1, label_1):
        feature_all = feature_1
        label_all = label_1
        proxies = F.normalize(self.proxies, p=2, dim=-1)
        feature_all = F.normalize(feature_all, p=2, dim=-1)

        D_ = torch.cdist(feature_all, proxies) ** 2

        mrg = torch.zeros_like(D_)
        mrg[label_all == 1] = mrg[label_all == 1] + self.mrg
        D_ = D_ + mrg

        p_loss = torch.sum(-label_all * F.log_softmax(-D_, 1), -1).mean()

        d_loss = self.cross_entropy(predict_1, torch.argmax(label_1, -1))

        loss = p_loss + d_loss
        return loss


