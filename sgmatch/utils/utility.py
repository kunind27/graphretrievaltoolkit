import os
from torch import cuda

def cudavar(x):
    return x.cuda() if cuda.is_available() else x