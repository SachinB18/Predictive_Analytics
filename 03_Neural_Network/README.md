# Neural Network Implementation - Predictive Analytics Assignment

## Overview

This project implements a complete **Neural Network** from scratch with:
- ✅ **Perceptron** - Single neuron building block
- ✅ **Feedforward Network** - Multi-layer neural network (MLP)
- ✅ **Backpropagation** - Training algorithm
- ✅ **Multiple Activation Functions** - Sigmoid, ReLU, Tanh, Softmax, Linear

## Project Structure

```
PA_Ass_3/
├── neural_network.py              # Core implementation
├── examples.py                    # Standalone examples
├── Neural_Network_Tutorial.ipynb  # Interactive Jupyter notebook
├── README.md                      # This file
└── THEORY.md                      # Mathematical theory
```

## Files Description

### 1. `neural_network.py`
Core neural network implementation with:
- `ActivationFunctions` - Collection of activation functions
- `Perceptron` - Single neuron
- `Layer` - Fully connected layer
- `FeedforwardNetwork` - Complete network with backpropagation

### 2. `Neural_Network_Tutorial.ipynb`
Interactive Jupyter notebook with:
- Step-by-step implementation
- **Iris Dataset** classification
- Training visualization
- Performance evaluation
- Architecture comparison
- Activation function testing

### 3. `examples.py`
Standalone Python examples:
- Example 1: XOR Problem
- Example 2: Binary Classification
- Example 3: Regression (Sine Wave)
- Example 4: Activation Functions Comparison
- Example 5: Learning Rate Impact
- Example 6: Network Architecture Impact

## Quick Start

### Option 1: Interactive Jupyter Notebook
```bash
cd PA_Ass_3
jupyter notebook Neural_Network_Tutorial.ipynb
```

### Option 2: Run Examples
```bash
cd PA_Ass_3
python examples.py
```

### Option 3: Use in Your Code
```python
from neural_network import FeedforwardNetwork
import numpy as np

# Create network: 2 inputs -> 4 hidden -> 1 output
network = FeedforwardNetwork(
    layer_sizes=[2, 4, 1],
    activations=['sigmoid', 'sigmoid']
)

# Train
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

network.train(X, y, epochs=1000, learning_rate=0.5)

# Predict
predictions = network.predict(X)
```

## Key Concepts

### 1. Perceptron
A single neuron that performs:
- **Forward Pass**: `z = w·x + b`, `a = activation(z)`
- **Backward Pass**: Update weights based on gradients

```
Input (x) ──→ [Weights] ──→ [Bias] ──→ [Activation] ──→ Output (a)
                                           ↑
                                    (Backprop gradients)
```

### 2. Layer
Multiple perceptrons working in parallel with matrix operations:
- **Forward**: `Z = X·W + b`, `A = activation(Z)`
- **Backward**: Compute gradients and update weights

### 3. Feedforward Network
Stack multiple layers to create a multi-layer perceptron (MLP):
```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
   (4)            (8)             (4)              (3)
```

### 4. Backpropagation Algorithm
Training algorithm that:
1. Forward propagates input through network
2. Computes loss at output
3. Backward propagates gradients through layers
4. Updates all weights and biases

## Activation Functions

| Function | Formula | Usage | Derivative |
|----------|---------|-------|-----------|
| **Sigmoid** | 1/(1+e^(-x)) | Hidden, Output (binary) | σ(x)·(1-σ(x)) |
| **ReLU** | max(0, x) | Hidden layers | 1 if x>0, else 0 |
| **Tanh** | (e^x - e^(-x))/(e^x + e^(-x)) | Hidden layers | 1 - tanh²(x) |
| **Softmax** | e^x_i / Σe^x_j | Output (multi-class) | - |
| **Linear** | x | Regression | 1 |

## Loss Functions

- **Cross-Entropy Loss** - Classification
  ```
  Loss = -Σ(y_true · log(y_pred))
  ```

- **Mean Squared Error (MSE)** - Regression
  ```
  Loss = Σ(y_pred - y_true)² / n
  ```

## Dataset Used

**Iris Dataset** (Classic machine learning benchmark):
- **Samples**: 150
- **Features**: 4 (Sepal length, width, Petal length, width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Task**: Multi-class classification

## Example Results

### Network Architecture: [4, 16, 8, 3]
```
Training Accuracy:  0.9833
Testing Accuracy:   0.9667
```

### Confusion Matrix:
```
                 Setosa  Versicolor  Virginica
Setosa               10            0          0
Versicolor            0            9          1
Virginica             0            0         10
```

## Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **learning_rate** | 0.1 | 0.01-1.0 | Speed of convergence |
| **epochs** | 100 | 50-1000 | Training iterations |
| **batch_size** | Full | 1-Full | Mini-batch training |
| **activation** | sigmoid | sigmoid, relu, tanh | Network expressiveness |

## Mathematical Background

### Forward Propagation
For each layer:
$$Z^{(l)} = A^{(l-1)} \cdot W^{(l)} + b^{(l)}$$
$$A^{(l)} = g^{(l)}(Z^{(l)})$$

Where:
- $Z^{(l)}$ = Pre-activation (linear combination)
- $A^{(l)}$ = Post-activation
- $W^{(l)}$ = Weight matrix
- $b^{(l)}$ = Bias vector
- $g^{(l)}$ = Activation function

### Backpropagation
Compute gradients from output to input:
$$\delta^{(l)} = \delta^{(l+1)} \cdot W^{(l+1)T} \odot g'^{(l)}(Z^{(l)})$$

Update weights:
$$W^{(l)} := W^{(l)} - \alpha \cdot \frac{\partial L}{\partial W^{(l)}}$$
$$b^{(l)} := b^{(l)} - \alpha \cdot \frac{\partial L}{\partial b^{(l)}}$$

## Performance Optimization

1. **Learning Rate**: Tune for faster convergence
2. **Batch Size**: Balance between speed and stability
3. **Network Depth**: More layers for complex patterns
4. **Activation Functions**: ReLU typically trains faster
5. **Feature Normalization**: Improves convergence

## References

1. **Rumelhart et al. (1986)** - Backpropagation Learning
2. **Goodfellow et al. (2016)** - Deep Learning Book
3. **LeCun et al. (1998)** - Efficient BackProp
4. **Glorot & Bengio (2010)** - Understanding Training Difficulty

## Requirements

- numpy >= 1.19.0
- pandas >= 1.1.0
- scikit-learn >= 0.23.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Author

Predictive Analytics Course Assignment
Academic Purpose - Step-by-step Neural Network Implementation

## License

MIT License - Feel free to use and modify for educational purposes.
