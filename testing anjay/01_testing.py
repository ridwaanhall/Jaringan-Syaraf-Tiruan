import numpy as np

# Features and labels
features = np.array([
    [1, 1, 1],
    [1, -1, 1],
    [-1, 1, 1],
    [-1, -1, 1]
])

labels = np.array([
    1,
    -1,
    -1,
    -1
])

# Initialize weights (including bias term)
w = [0.6, 0.2, -0.9]
learning_rate = 0.5
epochs = 10

# Training loop
for j in range(epochs):
    print('Epoch: ', j)
    global_delta = 0
    for i in range(features.shape[0]):

        actual = labels[i]
        instance = features[i]

        x0 = instance[0]
        x1 = instance[1]
        bias = instance[2]

        sum_unit = (w[0] * x0) + (w[1] * x1) + (w[2] * bias)

        if sum_unit > 0:
            prediction = 1
        else:
            prediction = -1

        delta = actual - prediction
        global_delta += abs(delta)

        print(f'Prediction: {prediction} (Actual: {actual}), Error: {delta}')
        w[0] += delta * learning_rate * x0
        w[1] += delta * learning_rate * x1
        w[2] += delta * learning_rate * bias

    print('-' * 20)

    if global_delta == 0:
        break

print("Final Weights:", w)
