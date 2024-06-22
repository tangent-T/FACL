import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Final_FC(nn.Module):
    def __init__(self, input_dim = 512, gost = 11+11, num_class = 120):
        super(Final_FC, self).__init__()

        self.fc = nn.Linear(input_dim * gost * 1, num_class)
        
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
    
    def forward(self, x):
        # x = x.reshape(1, -1)
        x= F.normalize(x, p=2, dim=1)
        prediction = self.fc(x)
        return prediction