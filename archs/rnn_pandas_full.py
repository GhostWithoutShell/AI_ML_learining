import pandas as pd
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import re
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('D:\MyFiles\Datasets\IMDB\IMDB Dataset.csv')
print("Data loaded:")
print(data.head())

def labelFix(dt):
    if dt == 'positive':
        return 1
    else:
        return 0

data['label'] = data['sentiment'].apply(labelFix) 
data = data.drop(columns='sentiment')

def removeHtmlTags(string):
    s2 = re.sub(r"<.*?>", "", string)
    return s2

data['review'] = data['review'].apply(removeHtmlTags)
print("\nData after preprocessing:")
print(data.head())

# Tokenizer and vocabulary
tokenizer = get_tokenizer("basic_english")

def generator_token(dt, tokenizer):
    for text in dt:
        yield tokenizer(text)

gen = generator_token(data['review'], tokenizer = tokenizer)
vocab = build_vocab_from_iterator(gen, specials=['<unk>', '<pad>'], max_tokens = 10000)
vocab.set_default_index(vocab["<unk>"])
stoi = vocab.get_stoi()  # dict: token -> int

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)

print("Default index:", vocab.get_default_index())           # Должен быть индекс <unk>
print("Index of <pad>:", vocab["<pad>"])                     # Например, 1
print("Index of unknown token:", vocab["abracadabraboom"]) 

# Dataset class
class IMDbDataset(Dataset):
    def __init__(self, input_ids_list, labels_list):
        self.inputs = input_ids_list
        self.labels = labels_list

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        return input_tensor, label_tensor

# Note: This code expects that 'input_ids' column exists in data
# The notebook had commented out preprocessing code for input_ids
# You may need to uncomment and implement the pad_and_encode function

# Split data
train_, test_ = train_test_split(data, test_size=0.2, random_state=32)

# Create datasets and dataloaders
train_dataset = IMDbDataset(train_['input_ids'].tolist(), train_['label'].tolist())
test_dataset = IMDbDataset(test_['input_ids'].tolist(), test_['label'].tolist())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check data shapes
for batch in train_loader:
    x, y = batch
    print("Train batch shapes:", x.shape, y.shape)
    break

for batch in test_loader:
    x, y = batch
    print("Test batch shapes:", x.shape, y.shape)
    break

# Model definition
class CustomRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, num_layers=2, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.h2o = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, batch):
        x = self.embedding(batch)
        rnn_out, (hidden, _) = self.rnn(x)
        last_hidden_forward = hidden[-1]
        last_hidden_backward = hidden[-2]
        last_hidden = torch.cat((last_hidden_backward, last_hidden_forward), dim=1)
        last_hidden = self.dropout(last_hidden)
        last_hidden = self.norm(last_hidden)
        logit = self.h2o(last_hidden)
        return logit

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomRNN(len(vocab), 128, 256).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10
losses = []

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1).float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
    print(f'epoch {epoch}, loss : {loss.item()}')
    losses.append(loss.item())

# Plot training losses
plt.plot(range(0, 10), losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluation
all_preds = []
all_labels = []
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1).float() 
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        correct += (preds == labels.bool()).sum().item()  
        total += labels.size(0)
        
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "rnn_88_lstm_weights.pth")

# Confusion matrix
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Data analysis
print("\nData statistics:")
print(data.count())

negative = data.loc[data['label'] == 0]['review']
print("\nNegative reviews length stats:")
print(negative.str.len().describe())

pos = data.loc[data['label'] == 1]['review']
print("\nPositive reviews length stats:")
print(pos.str.len().describe())

# Length distribution by class
data['length'] = data['review'].apply(lambda x: len(tokenizer(x)))
data.groupby('label')['length'].plot(kind='hist', alpha=0.5, legend=True)
plt.title("Длина отзывов по классам")
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

print("\nModel embedding layer:")
print(model.embedding)