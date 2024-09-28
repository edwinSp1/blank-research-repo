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
import atexit

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# input constants
ROWS = 10
COLS = 10
TRAIN_SAMPLES = 100000
TEST_SAMPLES = 20000
VALIDATION_SAMPLES = 20
HIDDEN_RATIO = 10

# If this is set to False, the program will try to get the model checkpoint from OUTPUT_PATH.
# otherwise it will generate a new checkpoint from scratch
FIRST_TIME_RUNNING = True

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

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
epoch = 0

if not FIRST_TIME_RUNNING:
    # reload from checkpoint
    checkpoint = torch.load(OUTPUT_PATH, weights_only=False)
    epoch = 0

    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    test_set = checkpoint['test_set']
    train_set = checkpoint['train_set']

def validate():
    validation_set = create_testcases(ROWS, COLS, num_samples=VALIDATION_SAMPLES)
    total_loss, accuracy = test_loop(validation_set)

    return total_loss, accuracy

def test_loop(test_set):
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

        percentage = correct/len(test_set)*100
        return total_loss, percentage


if __name__ == '__main__':

    if FIRST_TIME_RUNNING:
        train_set = create_testcases(ROWS, COLS, TRAIN_SAMPLES)
        test_set = create_testcases(ROWS, COLS, TEST_SAMPLES)

    print("train set and test set finished generating")

    # save model on exit
    def exit():
        torch.save({
            "state_dict": net.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "train_set": train_set,
            "test_set": test_set
        }, OUTPUT_PATH)
        print("model checkpoint saved to file.")

    atexit.register(exit)
    while True:
        

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

        # test model
        total_loss, accuracy = test_loop(test_set)
        ic(epoch)
        ic(total_loss)
        ic(accuracy)

        epoch += 1

