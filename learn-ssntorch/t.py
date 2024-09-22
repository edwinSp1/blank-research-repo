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

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.tensor([0.9, 0.1], dtype=torch.float)
target = torch.tensor([1, 0], dtype=torch.float)
output = loss(input, target)
print(input, target, output)