import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Historical optimal data. (Data building is worth further exploration.)
df = pd.read_excel(r'Historical optimal data.xlsx')  

# input:[x, y, angle, u, v, u2, v2]-7 dimensions
X = df[['x', 'y', 'angle', 'u', 'v', 'u2', 'v2']].values

# output:g_value-1 dimension
y = df['g_value'].values

# Divide the data into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  

# MLP network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = MLP()
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# trainning
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    
    # forward
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor)
    test_loss = criterion(predicted, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
torch.save(model.state_dict(), 'mlp_model.pth')
print("Model parameters saved to 'mlp_model.pth'")



