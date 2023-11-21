# Example: perceptron learning - 1
# Suppose we want to train a perceptron to compute AND
'''
training set:
x1 = 1, x2 = 1 -> y = 1
x1 = 1, x2 = -1 -> y = -1
x1 = -1, x2 = 1 -> y = -1
x1 = -1, x2 = -1 -> y = -1

randomly, let:
w0 = -0.9
w1 = 0.6
w2 = 0.2

using these weights:
x1 = 1, x2 = 1: -0.9*1 + 0.6*1 + 0.2*1 = -0.1  -1 WRONG
x1 = 1, x2 = -1: -0.9*1 + 0.6*1 + 0.2*-1 = -0.5  -1 OK
x1 = -1, x2 = 1: -0.9*1 + 0.6*-1 + 0.2*1 = -1.3  -1 WRONG
x1 = -1, x2 = -1: -0.9*1 + 0.6*-1 + 0.2*-1 = -1.7  -1 OK

'''
# %%
import numpy as np

# %%
features = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

# %%
w = [-0.9, 0.6, 0.2]

# %%
labels = np.array([
    1,
    -1,
    -1,
    -1
])

# %%
# make code from 'using there weights:'
for i in range(features.shape[0]):
    instance = features[i]
    x0 = instance[0]
    x1 = instance[1]
    sum_unit = w[0] * x0 + w[1] * x1 + w[2]
    if sum_unit > 0:
        prediction = 1
    else:
        prediction = -1
    print(f'x1 = {x0}, x2 = {x1}: {sum_unit} -> {prediction}')
    