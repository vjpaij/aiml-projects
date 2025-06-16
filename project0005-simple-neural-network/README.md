### Description:

In this project, we implement a simple feedforward neural network from scratch in Python—without using any high-level frameworks like TensorFlow or PyTorch. This is a great way to understand the inner workings of neural networks, including forward propagation, loss calculation, and backpropagation using only NumPy.

This neural network:
- Has 2 input neurons, 1 hidden layer with 4 neurons, and 1 output neuron
- Learns to model the XOR pattern (a nonlinear problem)
- Is implemented using only NumPy—ideal for learning backpropagation math

### XOR Neural Network in NumPy

This simple neural network is implemented from scratch using NumPy to learn the XOR logic gate, which is not linearly separable. It demonstrates basic feedforward and backpropagation operations.

---

### Code Explanation

#### 1. **Imports**

```python
import numpy as np
```

We import NumPy for numerical operations and array manipulations.

---

#### 2. **Activation Function and Derivative**

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
```

The sigmoid function maps inputs to a (0, 1) range. Its derivative is used during backpropagation to calculate gradients.

---

#### 3. **Input and Output Data**

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
```

These are the 4 possible input combinations for XOR and their respective outputs.

---

#### 4. **Initialization**

```python
np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
b1 = np.random.uniform(size=(1, hidden_neurons))

W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))
```

* A fixed seed ensures reproducible results.
* We define the number of neurons per layer and initialize weights and biases with random values.

---

#### 5. **Training Loop**

```python
epochs = 10000
learning_rate = 0.1
```

The model is trained over 10,000 epochs using a learning rate of 0.1.

```python
for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Error Calculation
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Weights and Biases Update
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")
```

* **Forward pass** computes the network predictions.
* **Loss** is calculated using mean squared error.
* **Backpropagation** computes gradients.
* **Weights and biases** are updated using gradient descent.
* The loss is printed every 1000 epochs.

---

#### 6. **Final Output**

```python
print("\nFinal predictions after training:")
print(final_output.round(3))
```

After training, we print the final predictions (rounded to 3 decimal places).

---

### Summary

This script demonstrates how a simple neural network can be trained to learn the XOR function using just NumPy, covering the fundamental concepts of forward and backward propagation, weight updates, and activation functions.
