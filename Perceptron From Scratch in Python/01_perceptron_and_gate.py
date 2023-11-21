# and gate
# %%
import numpy as np

# %%
features = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# %%
labels = np.array([
    0,
    0,
    0,
    1
])

# %%
w = [0.9, 0.9]
threshold = 0.5
learning_rate = 0.1
epochs = 10 

# %%
for j in range(0, epochs):
    print('Epoch: ', j)
    global_delta = 0
    for i in range(0, features.shape[0]):
        
        actual = labels[i]
        instance = features[i]
        
        x0 = instance[0]
        x1 = instance[1]
        
        sum_unit = w[0] * x0 + w[1] * x1
        
        if sum_unit > threshold:
            prediction = 1
        else:
            prediction = 0
            
        delta = actual - prediction
        global_delta = global_delta + abs(delta)
            
        # print(f'{instance} -> {prediction}')
        print(f'prediction = {prediction} where actual label is {actual}. error = {delta}')
        # print('')
        
        w[0] = w[0] + delta * learning_rate
        w[1] = w[1] + delta * learning_rate
    print('-'*20)
    
    if global_delta == 0:
        break
    
# %%
w
# %%
