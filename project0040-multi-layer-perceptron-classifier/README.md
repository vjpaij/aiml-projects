### Description:

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural networks with one or more hidden layers. It can model complex relationships and is widely used for classification tasks. In this project, we implement an MLP classifier using Keras to classify the Iris dataset.

- Builds a fully connected neural network for multi-class classification
- Trains and validates an MLP using backpropagation
- Visualizes the accuracy curve to monitor performance

## Iris Classification with a Neural Network (MLP)

This project demonstrates how to build a simple Multi-Layer Perceptron (MLP) using TensorFlow and Keras to classify the Iris dataset. The steps include data preprocessing, model training, evaluation, and performance visualization.

### Dependencies

Before running the code, ensure the required libraries are installed:

```bash
pip install tensorflow scikit-learn matplotlib
```

### Code Overview

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
```

* **Libraries**:

  * `sklearn`: for loading the Iris dataset, preprocessing, and splitting the data.
  * `tensorflow.keras`: for building and training the neural network.
  * `matplotlib`: for visualizing model performance.

### Data Loading and Preprocessing

```python
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42)
```

* **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance.
* **LabelBinarizer**: Converts categorical target labels to one-hot encoded vectors.
* **train\_test\_split**: Splits the dataset into training and testing sets (70% train, 30% test).

### Building the Neural Network

```python
model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
```

* **Input layer**: Accepts 4 features from the dataset.
* **Hidden layers**: Two layers with ReLU activation functions.
* **Output layer**: Softmax for multi-class classification (3 classes).

### Compiling the Model

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

* **Optimizer**: Adam, which is efficient and commonly used.
* **Loss function**: Categorical cross-entropy for one-hot encoded targets.
* **Metrics**: Accuracy to monitor model performance.

### Training the Model

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    verbose=0,
    validation_split=0.2
)
```

* Trains the model for 100 epochs with a batch size of 8.
* Uses 20% of training data for validation.

### Evaluation

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")
```

* Evaluates the trained model on the test data and prints the test accuracy.

### Plotting the Training History

```python
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('MLP Classification Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plots training and validation accuracy over epochs to visualize learning progress.

---

This example illustrates how a basic neural network can effectively classify the Iris dataset with high accuracy. It's a great starting point for learning neural network classification tasks.
