# Libraries
import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Dataset
from sklearn.datasets import fetch_california_housing

# Load the California Housing Dataset
data = fetch_california_housing()
x = data.data
y = data.target

# Split the Data into traning(80%) and testing(20%)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Scale the features using StandaradScaler
scaler= StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert to Pytorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train=torch.tensor(y_train.reshape(-1,1),dtype=torch.float32)
y_test=torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

# Create Tensor Datasets
train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)

# Create DataLoaders with batch size of 64
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size= 64, shuffle= False)

# Define the Neural Network Archietecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(8,32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x= self.fc2(x)
        return x         

# Create an Instance of the Model
model = NeuralNetwork()

# Define the Loss function and Optimizer
loss_function=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

# Training Loop
num_epochs=100
losses=[]
for epoch in range(num_epochs):
    epoch_loss=0

    # Training Phase
    for x_batch, y_batch in train_loader:
        # Forward Pass
        predictions=model(x_batch)
        # Calculate Loss
        loss=loss_function(predictions,y_batch)
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        #Update Weights 
        optimizer.step()

        epoch_loss+=loss.item()

    # Average Loss per epoch
    avg_loss = epoch_loss/len(train_loader)
    losses.append(avg_loss)

    #Print progress every 10 sec
    if (epoch+1)%10==0:
        print(f"Epoch{epoch + 1 }/{num_epochs},Loss:{avg_loss:.4f}")

# Evaluate on Test Data 
model.eval()
test_loss=0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        predictions = model(x_batch)
        loss=loss_function(predictions,y_batch)
        test_loss+=loss.item()
avg_test_loss=test_loss/len(test_loader)
print(f"\n Average Test Loss:{avg_test_loss:.4f}")  

#Plot training Loss
plt.figure(figsize=(10,6))
plt.plot(losses, label='Training Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss(MSE)')
plt.title('Neural Network Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
plt.close()
