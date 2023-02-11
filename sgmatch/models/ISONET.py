from typing import List, int, Optional, Dict
import torch
from torch import nn

class ISONET(nn.Module):
    def __init__(self, args):
        super(ISONET, self).__init__()
        self.args = args

    def forward(self):
        pass