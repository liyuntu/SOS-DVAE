import torch.nn as nn
import torch.nn.functional as F
from sklearn import decomposition as dp
import numpy as np
import torch
      
#NMF decoder        
class decoder(nn.Module):
    def __init__(self, args):
        super(decoder, self).__init__()
        
        W_init = self.softplus_inverse(args.comp)
        self.W_r = nn.Parameter(torch.FloatTensor(W_init).t(),requires_grad=True)

    def softplus_inverse(self, mat):
        return np.log(np.exp(mat) - 1 + 1e-8)

    def forward(self, x):
        #print(x.shape) #N*self.out_features
    
        W_un = F.softplus(self.W_r)
        
        W = F.normalize(W_un,dim=1,p=2) #where p=2 means the l2-normalization, and dim=1 means normalize tensor a with column.
        
        out = F.linear(F.softplus(x), W)

        return out