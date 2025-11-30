import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import roc_curve, auc

from noise_layers import AphexNoiseLayer as ap
from noise_layers import GaussianNoiseLayer as gn
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(58) 


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


class CIFAR10(nn.Module):
    def __init__(self, file_path, exprim_key=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        
        if exprim_key == 1:
            print("Using standard Dropout")
            self.drop1 = nn.Dropout(0.2)
            self.drop2 = nn.Dropout(0.2)
            self.drop3 = nn.Dropout(0.2)
        elif exprim_key == 2:
            print("Using Aphex Twin Noise Layer")
            
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.drop1 = ap(file_path, intensity=0.05, device=dev)
            self.drop2 = ap(file_path, intensity=0.05, device=dev)
            self.drop3 = ap(file_path, intensity=0.05, device=dev)
        elif exprim_key == 3:
            print("Using Gaussian Noise Layer")
            self.drop1 = gn(intensity=0.1)
            self.drop2 = gn(intensity=0.1)
            self.drop3 = gn(intensity=0.1)
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(64)
        
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU() 
        
        self.flat = nn.Flatten()
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.max_pool(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.max_pool(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.norm3(x)
        x = self.max_pool(x) 
        x = self.drop3(x)
        
        x = self.flat(x)
        x = self.linear(x)
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



train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, worker_init_fn=worker_init_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=worker_init_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


audio_file = "water.mp3" 


# SWITCH : 1 = Dropout, 2 = AphexNoise, 3 = GaussianNoiseLayer
model = CIFAR10(file_path=audio_file, exprim_key=3).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_epoch = 10

history = {
    'train_loss': [],
    'val_acc': []
}


for epoch in range(num_epoch):
    model.train() 
    running_loss = 0.0 
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()       
        output = model(images)      
        loss = criterion(output, labels) 
        loss.backward()             
        optimizer.step()            
        
        running_loss += loss.item()
    
    
    epoch_loss = running_loss / len(train_loader)
    history['train_loss'].append(epoch_loss)
    
    print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {epoch_loss:.4f}")


model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')


precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
plt.style.use('ggplot')

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Confusion Matrix Heatmap
cf = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cf, annot=True, fmt="d", cmap='Blues')
plt.title(f'Confusion Matrix (Acc: {accuracy:.2f}%)')
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()