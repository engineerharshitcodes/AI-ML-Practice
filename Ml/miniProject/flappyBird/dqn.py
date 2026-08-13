import torch 
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self,state_dim=12,action_dim=2,hidden_dim=256):
        super(DQN,self).__init__()
        
        self.model=nn.Sequential(
            nn.Linear(state_dim,hidden_dim),    #DQN expects 2D
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
        )
    def forward(self,X):
        return self.model(X)
        
    