import torch.nn as nn
import torch.nn.functional as F
import torch

# Outputting the latent parameters of distribution from the input data.
class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        self.device = args.device
        self.in_features = args.img_width * args.img_height
        
        h1= [512]
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features,h1[0]),
            nn.BatchNorm1d(h1[0]),
            nn.ReLU()
        )
        self.mu = nn.Linear(h1[0], args.out_features) #out_features: A integer indicating the latent dimension.
        self.log_var = nn.Linear(h1[0], args.out_features)        

    def mlp_ff(self, x, weights):
        out = F.linear(x, weights['mlp.0.weight'], weights['mlp.0.bias']) 
        out = F.batch_norm(out, torch.zeros(out.data.size()[1]).to(self.device), torch.ones(out.data.size()[1]).to(self.device),
                               weights['mlp.1.weight'], weights['mlp.1.bias'],
                               training=True)
        out = F.relu(out)                               
        return out

    # sample from the distribution having latent parameters z_mu, z_log_var
    # reparameterize
    # z_mu: mean
    # z_log_var: log(sigma^2) = log(std^2)
    def sampling(self, z_mu, z_log_var):
        std = torch.exp(z_log_var / 2)  #exp(log(sigma^2)/2) = exp(log(sigma)) = sigma
        eps = torch.randn_like(std)     #a tensor with the same size as "std" that is filled with random numbers from a normal distribution with mean 0 and variance 1.
        z = eps.mul(std).add_(z_mu)     #z = mean + sigma*epsilon
        
        return z #N*100
    
    def forward(self, x, weights=None):
        if weights is None:
            hidden = self.mlp(x)
            z_mu = self.mu(hidden)
            z_log_var = self.log_var(hidden)
        else:
            hidden = self.mlp_ff(x, weights)
            z_mu  = F.linear(hidden, weights['mu.weight'],  weights['mu.bias'])
            z_log_var = F.linear(hidden, weights['log_var.weight'], weights['log_var.bias'])
        
        #latent variable z
        z = self.sampling(z_mu, z_log_var)
        
        return z_mu, z_log_var, z