import torch

import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,kernel_size=3,padding =1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3,padding =1)
        self.bN2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxPol1 = nn.MaxPool2d(2, 2)
        self.dr0 = nn.Dropout(0.2)
        self.maxPol2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3,padding =1)
        self.bN3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.dr1 = nn.Dropout(0.2)
        self.maxPol3 = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.lin = nn.Linear(62, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bN2(x)
        x = self.relu2(x)
        x = self.maxPol1(x)
        x = self.dr0(x)
        x = self.maxPol2(x)
        x = self.conv3(x)
        x = self.bN3(x)
        x = self.relu3(x)
        x = self.dr1(x)
        x = self.maxPol3(x)
        x = self.flat(x)
        #if self.lin is None:
        #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #    in_features = x.shape[1]
        #    self.lin = nn.Linear(in_features, 10).to(x.device)
        x = self.lin(x)
        return x
    



transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 25
losess = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        losess.append(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

print(f"Test Accuracy: {100 * correct / total:.2f}%")

import numpy as np
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
cm = confusion_matrix(all_labels, all_preds)
print(cm)

import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()