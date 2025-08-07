# 🧠 XOR Neural Network with PyTorch

A beginner-friendly neural network that learns to solve the classic **XOR problem**, built with **PyTorch** from scratch. This project is designed to help you deeply understand how neural networks work — neuron by neuron, layer by layer.

---

## 📌 What is XOR?

**XOR (Exclusive OR)** is a logic gate that outputs `1` when its two binary inputs are different, and `0` when they are the same.

| Input A | Input B | Output (A XOR B) |
|---------|---------|------------------|
|   0     |    0    |        0         |
|   0     |    1    |        1         |
|   1     |    0    |        1         |
|   1     |    1    |        0         |

XOR is **not linearly separable**, meaning you can’t draw a single straight line to separate the outputs. That’s why a neural network needs **at least one hidden layer** to learn XOR.

---

## 🧠 What Is a Neural Network?

A **neural network** is a machine learning model inspired by the human brain. It consists of:
- **Neurons (nodes)** that receive inputs
- **Weights** that scale those inputs
- **Biases** that shift the values
- **Activation functions** that add non-linearity

The network **learns** by adjusting the weights and biases to reduce the difference between predictions and actual values — a process called **training**.

---

## 🏗️ Architecture of Our XOR Model

```
Input Layer (2 neurons)
      ↓
Hidden Layer (4 neurons, Sigmoid)
      ↓
Output Layer (1 neuron, Sigmoid)
```

### Components:
- **Input Layer:** 2 neurons (for A and B)
- **Hidden Layer:** 4 neurons with Sigmoid activation
- **Output Layer:** 1 neuron with Sigmoid activation
- **Loss Function:** Mean Squared Error
- **Optimizer:** Stochastic Gradient Descent (SGD)

---

## 📜 Full Code Walkthrough

### 1. 🔧 Imports and Data

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# XOR input and output data
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
```

- **X:** Input combinations (4 examples)
- **Y:** Expected XOR outputs

---

### 2. 🧠 Model Definition

```python
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x
```

- **Layer 1:** Connects 2 inputs → 4 hidden neurons
- **Layer 2:** Connects 4 → 1 output
- **Activation Function:** Sigmoid squashes outputs between 0 and 1

---

### 3. 🎓 Loss and Optimizer

```python
model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

- **Loss:** Measures how wrong the prediction is
- **SGD Optimizer:** Adjusts weights to reduce loss
- **Learning rate (lr=0.1):** Controls how fast the model learns

---

### 4. 🔁 Training Loop

```python
losses = []
for epoch in range(5000):
    output = model(X)
    loss = criterion(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

- **Forward Pass:** Generates prediction
- **Backward Pass:** Computes gradients
- **Step:** Updates weights
- Trains for 5000 epochs (rounds)

---

### 5. 📉 Plotting the Loss Curve

```python
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```

- Visualizes how the model improves over time

---

### 6. ✅ Final Predictions

```python
with torch.no_grad():
    predictions = model(X).round()
    print("\nFinal Predictions:")
    print(predictions)
```

- Uses the trained model to predict XOR values
- `.round()` converts probabilities (e.g. 0.98) to 1 or 0

---

## ▶️ How to Run the Project

### 📦 1. Install Requirements

```bash
pip install -r requirements.txt
```

### ▶️ 2. Run the Script

```bash
python xor_nn.py
```

You’ll see:
- Loss values decreasing
- A plotted loss curve
- Final predictions printed as output

---

## 📁 Project Files

| File          | Purpose                               |
|---------------|----------------------------------------|
| xor_nn.py     | Full code of the neural network        |
| README.md     | Detailed explanation of the project    |
| requirements.txt | Python libraries needed             |
| .gitignore    | Optional: ignores unnecessary files    |

---

## 📚 Concepts You’ll Learn

- How neural networks process inputs
- Forward and backward propagation
- Activation functions (Sigmoid)
- Loss functions and optimizers
- Training loops in PyTorch

---

## 🚀 Future Upgrades (Optional Ideas)

- Replace Sigmoid with **ReLU** or **Tanh**
- Add more hidden layers or neurons
- Try **Adam** optimizer instead of SGD
- Train on other logic gates (AND, OR, NAND)
- Turn into a web app with Streamlit

---

## 💬 Credits & Author

Made by [Your Name or GitHub handle]  
Beginner-level project to understand PyTorch, logic, and training basics

---

## ⭐ Star This Repo

If you learned something from this project, consider starring ⭐ it to help others discover it!
