import numpy as np
import itertools
import torch
def has_line(mat):
    for row in range(3):
        # horizontal line is all ones
        if sum(mat[row]) == 3:
            return True
    
    for col in range(3):
        s = 0
        for row in range(3):
            s += mat[row][col]
        # vertical line
        if s == 3:
            return True
    
    return False

def create_testcases():
    train_set = []
    # iterate through all subsets
    for mask in range(1<<9):
        
        mat = [[0 for i in range(3)] for j in range(3)]
        
        for i in range(9):
            # SAMPLE: 
            # mask: 101101 base 2
            # i: 2
            # 101101 >> 2 == 1011
            # 1011 & 1 = 1
            # so ith bit of mask is set to one

            bit = (mask >> i) & 1
            mat[i//3][i%3] = bit

        flattened = []
        for row in mat:
            for x in row:
                flattened.append(x)
        train_set.append((torch.tensor(np.array(flattened), dtype=torch.float), has_line(mat)))
    
    return train_set

if __name__ == '__main__':
    test_set = create_testcases()
    from icecream import ic
    ic(test_set)

    



