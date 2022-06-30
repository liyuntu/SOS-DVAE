import torch.nn as nn

# The decoder takes a sample from the latent dimension and uses that as an input to generate X.        
class decoder(nn.Module):
    def __init__(self, args):
        super(decoder, self).__init__()
        
        self.in_features = args.img_width * args.img_height
        
        h1= [512]
        self.mlp = nn.Sequential(
            nn.Linear(args.out_features,h1[0]),
            nn.ReLU(),
            nn.Linear(h1[0], self.in_features),
            nn.ReLU()
        ) 
        
    def forward(self, x):
        return self.mlp(x)