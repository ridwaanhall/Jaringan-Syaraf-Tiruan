import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.5, epochs=100):
        # Initialize weights and bias
        self.weights = np.array([1, -0.9, 0.2])
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        # Compute the weighted sum and add bias
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else -1

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            all_correct = True
            for inputs, label in zip(training_inputs, labels):
                # Compute the weighted sum
                weighted_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]

                # Apply the activation function
                prediction = 1 if weighted_sum > 0 else -1

                # Update weights and bias
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

                # Check if prediction is correct
                if prediction != label:
                    all_correct = False

                # Print intermediate steps
                print(f"Epoch {epoch + 1}, Inputs: {inputs}, Weighted Sum: {weighted_sum:.5f}, Prediction: {prediction}, Target: {label}, Weights: {self.weights}")

            # If all predictions are correct, stop training
            if all_correct:
                print("Training complete. All predictions are correct.")
                break

# Define training data for the AND gate with -1 and 1 labels
training_inputs = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

labels = np.array([1, -1, -1, -1])

# Create and train the perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.5, epochs=100)
perceptron.train(training_inputs, labels)
