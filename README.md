# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: SHABREENA VINCENT

### Register Number: 212222230141

```
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Dataset
np.random.seed(42)
torch.manual_seed(42)

X = np.arange(1, 51, dtype=np.float32).reshape(-1, 1)
noise = np.random.normal(0, 5, X.shape)
y = 2 * X + 3 + noise  # y = 2x + 3 with noise

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Step 2: Define the Model
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.linear(x)

model = Model(1, 1)

# Step 3: Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Step 4: Train the Model
epochs = 100
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Step 5: Plot Loss Curve
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Iterations')
plt.show()

# Step 6: Visualize the Best-Fit Line
predicted = model(X_tensor).detach().numpy()
plt.scatter(X, y, label='Original Data')
plt.plot(X, predicted, color='red', label='Best Fit Line')
plt.legend()
plt.title('Linear Regression Model')
plt.show()

# Step 7: Make Predictions
new_input = torch.tensor([[55.0]])  # Predict for X=55
predicted_value = model(new_input).item()
print(f'Prediction for X=55: {predicted_value:.4f}')

```

### Dataset Information
Include screenshot of the generated data

### OUTPUT
Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/0b04c4ac-5864-495e-9b24-c0b8d6490644)

Best Fit line plot

![image](https://github.com/user-attachments/assets/27deb3af-162c-4367-af69-8f447d6b5de9)

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
