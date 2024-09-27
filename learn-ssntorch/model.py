# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.functional import normalize

import matplotlib.pyplot as plt
import numpy as np
import itertools

from create_testcases import create_testcases
from icecream import ic

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# input constants
ROWS = 10
COLS = 10
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 2000
HIDDEN_RATIO = 10

# output constants
OUTPUT_PATH = 'models/10x10Recognizer'
# Network Architecture
num_inputs = ROWS*COLS
num_hidden = ROWS*COLS*HIDDEN_RATIO
num_outputs = 2

# Temporal Dynamics
num_steps = 25
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        
    def forward(self, x):
        
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        # Record the final layer
        mem2_rec = []
        spk2_rec = []
        for step in range(num_steps):
            # linear transformation 1
            cur1 = self.fc1(x)
            # apply leaky
            spk1, mem1 = self.lif1(cur1, mem1)
            # linear transformation 2
            cur2 = self.fc2(spk1)
            # apply leaky again
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2_rec.append(mem2)
            spk2_rec.append(spk2)
        
        return torch.stack(mem2_rec, dim=0), torch.stack(spk2_rec, dim=0)

# Load the network onto CUDA if available
net = Net().to(device)

if __name__ == '__main__':
    train_set = create_testcases(ROWS, COLS, TRAIN_SAMPLES)
    test_set = create_testcases(ROWS, COLS, TEST_SAMPLES)

    print("train set and test set finished generating")
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    num_epochs = 100


    for epoch in range(1, num_epochs+1):
        for matrix_hash, label in train_set:
            # training mode
            net.train()
            
            # put the tensors onto CUDA
            matrix_hash = matrix_hash.to(device)
            label = label.to(device)

            # get model prediction
            mem_rec, spk_rec = net(matrix_hash)

            # calculate loss
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        
        with torch.no_grad():
            correct = 0
            total_loss = 0
            for matrix_hash, label in test_set:
                # put the tensors onto CUDA
                matrix_hash = matrix_hash.to(device)
                label = label.to(device)

                # get model prediction
                mem_rec, spk_rec = net(matrix_hash)
                
                # calculate loss
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                amt = torch.zeros((2), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], label)
                    amt += mem_rec[step]
                
                amt_array = list(amt.numpy())
                label_array = list(label.numpy())
                ans = 0
                if label[1] == 1:
                    ans = 1
                # higher probability than wrong answer
                if amt_array[ans] > amt_array[ans^1]:
                    correct += 1
                total_loss += loss_val

            ic(epoch)
            ic(total_loss)
            percentage = correct/len(test_set)*100
            print('accuracy:', f'{percentage}%')
            # 99.5% is good enough
            if percentage > 99.5:
                break

    torch.save(net.state_dict(), OUTPUT_PATH)