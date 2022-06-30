import torch.nn as nn

class classifier(nn.Module):
    def __init__(self, args):
        super(classifier, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(args.out_features, args.num_labels)        
        )
    
    def forward(self,x):
        return self.mlp(x)