import numpy as np
import itertools
import torch
import random

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
""" # TODO
def rand_mat(rows, cols):
    mat = [[0 for i in range(cols)] for j in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if random.random()
"""

def create_testcases(rows, cols, num_samples):
    train_set = []
    # iterate through all subsets
    for _ in range(num_samples):
        rand_mat = torch.rand((rows, cols), dtype=torch.float)
        mat = torch.round(rand_mat)

        # convert label to probability tensor
        label = [0, 0]
        label[has_line(mat)] = 1
        
        train_set.append((torch.flatten(mat), torch.tensor(label, dtype=torch.float)))
    
    return train_set

if __name__ == '__main__':
    test_set = create_testcases(3, 3, 10)
    from icecream import ic
    ic(test_set)

    



