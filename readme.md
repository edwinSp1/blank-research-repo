Research Repository for RC and snnTorch.

My plan is to expand to 10x10 after I implement the 3x3 with Reservoir Computing.

To avoid overtraining, I will probably use a test set and a train set. If the model performs well on the training set but poorly on the test set I know it's overtrained. The reason my model does tests on the train set is because the train set is all 2^9 possible 3*3 matrices.

The input is a tensor representing the flattened matrix, ex: tensor([1, 0, 0, 1, 1, 0, 1, 0, 1])

It is then transformed similarly to visualizations/splt.animator.mp4.

The output is a 2D Tensor of confidences, where tensor[i] = [confidence_no_line, confidence_line]

