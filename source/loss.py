import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    
    def __init__(self,*kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self,pred, label):
        return ...