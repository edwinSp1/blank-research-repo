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

from create_testcases import create_testcases

# dataloader arguments
batch_size = 128
data_path='/tmp/data/mnist'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])


train_set = create_testcases()

# Network Architecture
num_inputs = 3*3
num_hidden = 100
num_outputs = 2

# Temporal Dynamics
num_steps = 25
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers

        self.fc1 = nn.Linear(num_inputs, num_hidden),
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs),
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    
# Load the network onto CUDA if available
net = Net().to(device)

# look. nn.Linear() isn't supposed to return a tuple,
# BUT YET IT DOES
# so we have to fix that... for some reason
net.fc1 = net.fc1[0]
net.fc2 = net.fc2[0]

input, ans = train_set[0]
print(input, ans)
print(input.size())
print(net(input))
print(net(input))
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 1

"""
for epoch in range(num_epochs):
    for matrix, label in train_set:
        matrix = torch.tensor(matrix)
        print(matrix)
"""