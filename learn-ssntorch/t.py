# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

m = nn.Linear(3*3, 3)
print(m)
input = torch.randn(128, 20)
output = m(input)
print(output.size())