import math
import numpy as np
import itertools
import torch
import random
from icecream import ic
def has_line(mat):
    n = len(mat)
    m = len(mat[0])
    for row in range(n):
        # horizontal line is all ones
        if sum(mat[row]) == m:
            return True
    
    for col in range(m):
        s = 0
        for row in range(n):
            s += mat[row][col]
        # vertical line
        if s == n:
            return True
    
    return False

# modifies mat IN PLACE 
# and returns it
# if mat[i][j] is one then it won't do anything
# else it randomly picks it
def fill_remaining(mat):
    rows = len(mat)
    cols = len(mat[0])
    for i in range(rows):
        for j in range(cols):
            if mat[i][j] == 1:
                continue
            mat[i][j] = round(random.random())
    return mat 

# modifies in place
def add_line(mat, i, type="horizontal"):
    if type == 'horizontal':
        for j in range(len(mat[0])):
            mat[i][j] = 1
    else:
        for j in range(len(mat)):
            mat[j][i] = 1
    
    return mat


def create_testcases(rows, cols, num_samples):
    train_set = []
    amt_has_line = 0
    # iterate through all subsets
    for _ in range(num_samples):
        mat = [[0 for i in range(rows)] for j in range(cols)]

        fill_remaining(mat)

        # convert label to probability tensor
        label = [0, 0]
        label[has_line(mat)] = 1
        if label[1]:
            amt_has_line += 1

        train_set.append(
            (
                torch.flatten(torch.tensor(mat, dtype=torch.float)), 
                torch.tensor(label, dtype=torch.float)
            )
        )

    # imbalanced dataset
    if num_samples//2 > amt_has_line:
        for _ in range(num_samples//2 - amt_has_line):
            mat = [[0 for i in range(rows)] for j in range(cols)]

            # put a random amount of lines into the matrix
            for line in range(random.randint(1, max(1, rows//3))):
                if random.random() < 0.5:
                    add_line(mat, random.randint(0, rows-1), type="horizontal")
                if random.random() < 0.5:
                    add_line(mat, random.randint(0, cols-1), type="vertical")
            
            fill_remaining(mat)

             # convert label to probability tensor
            label = [0, 0]
            label[has_line(mat)] = 1
            train_set.append(
                (
                    torch.flatten(torch.tensor(mat, dtype=torch.float)), 
                    torch.tensor(label, dtype=torch.float)
                )
            )

    # should be more balanced now, shuffle it and take num_samples out 
    random.shuffle(train_set) 
    return train_set[:num_samples]

if __name__ == '__main__':
    test_set = create_testcases(3, 3, 1000)
    count_has_line = 0
    for mat, label in test_set:
        if label[1]:
            count_has_line += 1
    #ic(test_set)
    ic(count_has_line/len(test_set))

    



