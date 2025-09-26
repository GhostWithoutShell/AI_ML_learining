import sys
print(sys.executable)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

class CustomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.lin0 = nn.Linear(784, 128)
        self.rel0 = nn.ReLU()
        self.lin1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.flat(x)
        x = self.lin0(x)
        x = self.rel0(x)
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        return x

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# Load datasets
train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform_train)
test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform_test)
print(len(test_ds))

# Create data loaders
train_ds = DataLoader(train_ds, batch_size=64, shuffle=False)
test_ds = DataLoader(test_ds, batch_size=64, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss function and model setup
crptron = torch.nn.CrossEntropyLoss()
model = CustomMLP().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epoch = 15
losses = []
for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for images, labels in train_ds:
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(images)
        loss = crptron(outputs, labels)
        
        losses.append(loss)
        loss.backward()
        optim.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {running_loss/len(train_ds):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_ds:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Confusion matrix
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# Heatmap visualization
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.show()

# Save model
torch.save(model.state_dict(), "mlp_model.pth")

# Test loading model
del model
model = CustomMLP()
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()