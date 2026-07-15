# Artificial Neural Network From Scratch (NumPy)

A fully connected Artificial Neural Network (ANN) implemented **from scratch using only NumPy**.

This project demonstrates the complete mathematical workflow behind neural networks without relying on deep learning frameworks such as TensorFlow or PyTorch.

The implementation covers every major component of a feed-forward neural network, including forward propagation, backpropagation, gradient descent, loss computation, and prediction.

---

## Features

- Built entirely with **NumPy**
- Dynamic network architecture
- Forward Propagation
- Backward Propagation
- Gradient Descent Optimization
- Binary Cross Entropy Loss
- Mean Squared Error Loss
- Multiple Activation Functions
- Prediction & Accuracy Evaluation
- XOR classification example

---

## Network Architecture

The network architecture is configurable.

Example:

```python
layer_sizes = [2, 4, 1]
```

which represents

```
Input Layer (2)
      │
Hidden Layer (4, ReLU)
      │
Output Layer (1, Sigmoid)
```

The implementation also supports deeper architectures such as

```python
layer_sizes = [2, 8, 6, 4, 1]
```

without modifying the forward or backward propagation logic.

---

## Mathematical Pipeline

```
Input Data
     │
Normalization
     │
Weight Initialization
     │
Forward Propagation
     │
Loss Computation
     │
Backward Propagation
     │
Gradient Descent
     │
Updated Parameters
     │
Prediction
```

---

## Implemented Components

### Data Preprocessing

- Min-Max Normalization

---

### Weight Initialization

Random Gaussian initialization

```
W ~ N(0, 0.1)
b = 0
```

---

### Activation Functions

- Sigmoid
- ReLU
- Tanh
- Softmax

Derivative implementations are included for backpropagation.

---

### Loss Functions

- Binary Cross Entropy (BCE)
- Mean Squared Error (MSE)

---

### Optimization

- Gradient Descent

Parameter update rule

```
W = W - α dW
b = b - α db
```

---

## Project Structure

```
ANN_From_Scratch.py

├── Data Normalization
├── Weight Initialization
├── Activation Functions
├── Forward Propagation
├── Loss Functions
├── Backward Propagation
├── Gradient Descent
├── Training Loop
├── Prediction
└── XOR Example
```

---

## Example Output

```
Epoch 0/10000 — Loss: ...

Epoch 2000/10000 — Loss: ...

Epoch 4000/10000 — Loss: ...

...

Accuracy: 100%
```

---

## Example Problem

The network is trained on the classic XOR problem.

```
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

Since XOR is **not linearly separable**, it serves as a standard benchmark for neural networks.

---

## Technologies

- Python
- NumPy

---

## Learning Objectives

This project was created to gain a deeper understanding of

- Neural Network Mathematics
- Matrix Operations
- Forward Propagation
- Backpropagation
- Chain Rule
- Gradient Descent
- Binary Classification

without relying on high-level deep learning libraries.

---

## Future Improvements

- Xavier Initialization
- He Initialization
- Adam Optimizer
- RMSProp
- Momentum
- Dropout
- L2 Regularization
- Softmax + Cross Entropy
- Mini-Batch Gradient Descent
- Multi-class Classification