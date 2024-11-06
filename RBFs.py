import numpy as np
import torch
import time
import gc
import copy

def RBF_Mult(r,c):
    phi=torch.sqrt(r+torch.square(c))
    return phi

def RBF_Gaussian(r,c):
    phi=torch.exp(-c*c*torch.square(r))
    return phi

def RBF_TPS(r):
    phi=torch.square(r)*torch.log(r)
    return phi