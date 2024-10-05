# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

from create_testcases import create_testcases
from icecream import ic
import atexit

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# input constants
ROWS = 28
COLS = 28
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 2000
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

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 2)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through ``fc1``
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      output = torch.flatten(output)
      return output

net = Net()

# Load the network onto CUDA if available
net = Net().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=5e-5, momentum=0.9)
epoch = 0

accuracy_data = []
if not FIRST_TIME_RUNNING:
    # reload from checkpoint
    checkpoint = torch.load(OUTPUT_PATH, weights_only=False)
    epoch = 0

    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    accuracy_data = checkpoint['accuracy_data'] 

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
            pred = net(matrix_hash)
            
            total_loss += loss(pred, label)
            ans = int(label[1])
            pred_arr = list(pred.numpy())
            if pred_arr[ans] > pred_arr[ans^1]:
                correct += 1

        percentage = correct/len(test_set)*100
        return total_loss, percentage


if __name__ == '__main__':

    #if FIRST_TIME_RUNNING:
    train_set = create_testcases(ROWS, COLS, TRAIN_SAMPLES)
    test_set = create_testcases(ROWS, COLS, TEST_SAMPLES)

    print("train set and test set finished generating")
    print(net(train_set[0][0]))
    # save model on exit
    def exit():
        torch.save({
            "state_dict": net.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            #"train_set": train_set,
            #"test_set": test_set
        }, OUTPUT_PATH)
        print("model checkpoint saved to file.")

    atexit.register(exit)
    while True:
        loss_val = 0
        correct = 0
        for matrix_hash, label in train_set:
            # put the tensors onto CUDA
            matrix_hash = matrix_hash.to(device)
            label = label.to(device)

            # get model prediction
            pred = net(matrix_hash)
            
            loss_val += loss(pred, label)
            
            pred_arr = list(int(x) for x in pred)
            label_arr = list(int(x) for x in label)
            ans = sum(label_arr)
            if pred_arr[ans] > pred_arr[ans^1]:
                correct += 1

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss = loss_val
        train_accuracy = correct/len(train_set) * 100
        # test model
        test_loss, test_accuracy = test_loop(test_set)
        ic(epoch)
        ic(train_loss)
        ic(train_accuracy)
        ic(test_loss)
        ic(test_accuracy)

        epoch += 1

