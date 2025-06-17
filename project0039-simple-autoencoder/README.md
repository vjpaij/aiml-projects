### Description:

An autoencoder is a type of artificial neural network used for learning efficient codings of input data in an unsupervised manner. It consists of two parts: an encoder that compresses the input and a decoder that reconstructs it. In this project, we build a simple autoencoder using Keras to compress and reconstruct the Iris dataset.

- Compresses 4D data to 2D using an autoencoder
- Learns a non-linear transformation
- Visualizes learned embeddings and structure between classes

### Autoencoder on Iris Dataset

This script demonstrates how to use a simple **autoencoder** built with TensorFlow and Keras to reduce the dimensionality of the classic **Iris dataset** for visualization purposes.

---

### 🧩 What is an Autoencoder?

An **autoencoder** is a type of neural network used to learn efficient data codings in an unsupervised manner. It consists of two parts:

* **Encoder**: Compresses the input into a smaller representation.
* **Decoder**: Reconstructs the input from the compressed representation.

---

### 📦 Installation

Before running the script, make sure to install the required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

### 📊 Code Explanation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
```

These are the essential imports:

* `Iris dataset` for input data
* `MinMaxScaler` to normalize features between 0 and 1
* `Keras` modules to build and train the autoencoder

```python
iris = load_iris()
X = iris.data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

Load the Iris dataset and normalize its features. Normalization improves neural network training performance.

```python
input_dim = X_scaled.shape[1]  # Number of input features (4 for Iris)
encoding_dim = 2              # Reduce to 2 dimensions for visualization
```

We want to reduce the 4D data to 2D.

#### 🏗 Define Autoencoder Architecture

```python
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
```

* `Input Layer`: Accepts 4-dimensional input.
* `Encoded Layer`: Compresses input to 2 dimensions.
* `Decoded Layer`: Reconstructs the 4-dimensional output from the 2D code.

```python
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
```

* The autoencoder model is built and compiled using Mean Squared Error loss.

```python
history = autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=16, verbose=0)
```

* The model is trained for 100 epochs. Both input and target are `X_scaled`, as the goal is to reconstruct the input.

#### 🔍 Extract Compressed Data

```python
encoder = Model(inputs=input_layer, outputs=encoded)
encoded_data = encoder.predict(X_scaled)
```

* This creates a new model (the encoder) to generate the 2D compressed representation of the data.

#### 📈 Visualize Encoded Data

```python
plt.figure(figsize=(7, 5))
for i in range(3):
    plt.scatter(encoded_data[iris.target == i, 0], encoded_data[iris.target == i, 1], label=iris.target_names[i])
plt.title("2D Encoded Representation (Autoencoder)")
plt.xlabel("Encoded Dimension 1")
plt.ylabel("Encoded Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plot the 2D encoded data, colored by class (`setosa`, `versicolor`, `virginica`).
* This shows how well the autoencoder separates different classes in the reduced space.

---

### ✅ Summary

* The autoencoder compresses the 4D Iris data into 2D.
* You can visualize how different Iris species cluster in the encoded space.
* This is a simple unsupervised learning technique for dimensionality reduction and data visualization.
