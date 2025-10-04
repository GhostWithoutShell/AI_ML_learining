import pandas as pd
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import json
import re


class SimpleAttention(nn.Module):
    def __init__(self, size_kernel):
        super().__init__()
        self.size = size_kernel
        self.key = nn.Linear(size_kernel, int(size_kernel/2))
        self.value = nn.Linear(size_kernel, int(size_kernel/2))
        self.query = nn.Linear(size_kernel, int(size_kernel/2))
    def forward(self, x):
        x_k = self.key(x)
        x_v = self.value(x)
        x_q = self.query(x)


        print(f"Shape v :{x_v.shape}")
        print(f"Shape q :{x_q.shape}")
        print(f"Shape k :{x_k.shape}")

        transpose_k = torch.transpose(x_k, -2, -1)
        
        attention_score = torch.matmul(x_q, transpose_k)
        print(f"Transpose {transpose_k.shape}")
        scaled_scores = attention_score/math.sqrt(self.size)
        print(f"Scaled scores {scaled_scores}, shape {scaled_scores.shape}")
        att_weight = torch.softmax(scaled_scores, dim=1)
        print(f"Att weight {att_weight}, shape {att_weight.shape}")
        result_mat = torch.matmul(att_weight, x_v)
        return torch.mean(result_mat, dim=1)


class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embeding_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embeding_dim)
        self.attention = SimpleAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size//2)
        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(hidden_size//2, 1)
    def forward(self, x):
        x = self.emb(x)
        x = self.attention(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.lin(x)
        return x
        

def labelFix(dt):
    if dt == 'positive':
        return 1
    else:
        return 0
    
def pad_and_encode(data, index_pad):
    tokens = tokenizer(data)
    indexes = [vocab[token] for token in tokens]
    if len(indexes) < 256:
        padding_length = 256 - len(indexes)
        indexes += [index_pad] * padding_length
    elif len(indexes) > 256:
        indexes = indexes[0:256]
    return indexes

#vocab creating pipline
tokenizer = get_tokenizer("basic_english")
data = pd.read_csv('D:\MyFiles\Datasets\IMDB\IMDB Dataset.csv')
data.columns = ['review', 'label']
data['label'] = data['label'].apply(labelFix)

def removeHtmlTags(string):
    s2 = re.sub(r"<.*?>", "", string)
    return s2

data['review'] = data['review'].apply(removeHtmlTags)

def gen_tokenizer(data, tokenizer):
    for text in data:
        yield tokenizer(text)

gen = gen_tokenizer(data['review'], tokenizer=tokenizer)
vocab = build_vocab_from_iterator(gen, specials=['<unk>', '<pad>'], max_tokens=10000)
vocab.set_default_index(vocab["<unk>"])
data["input_ids"] = data["review"].apply(pad_and_encode, index_pad=vocab["<pad>"])
stoi = vocab.get_stoi()
with open("vocab.json", "w", encoding = "utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)
print("Default index:", vocab.get_default_index())
print("Index of <pad>:", vocab["<pad>"])
print("Index test :", vocab["13dfsdafsf"])


#dataset object
class LabelsIdsDataset(Dataset):
    def __init__(self, input_id_list, label_list):
        self.inputs = input_id_list
        self.labels = label_list
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.inputs[index], dtype=torch.long)
        labels_tensor = torch.tensor(self.labels[index], dtype=torch.float)
        return input_tensor, labels_tensor

#split data

train_, test_ = train_test_split(data, test_size=0.3, random_state=45)
train_, valid_ = train_test_split(train_, test_size=0.2, random_state=32)

# prepare data for train 

train_ = LabelsIdsDataset(train_["input_ids"].tolist(), train_["label"].tolist())
test_ = LabelsIdsDataset(test_["input_ids"].tolist(), test_["label"].tolist())
valid_ = LabelsIdsDataset(valid_["input_ids"].tolist(), valid_["label"].tolist())

train_dataloader = DataLoader(train_, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_, batch_size=32, shuffle=False)
valid_dataloader = DataLoader(valid_, batch_size=32, shuffle=True)


# Быстрый тест без полного обучения
#for i, (inputs, labels) in enumerate(train_dataloader):
#    if i > 2:  # Только 3 батча для проверки
#        break
#    # твой код здесь

#check shapes of tokens

for batch in train_dataloader:
    x, y = batch
    print("Train batch shapes:", x.shape, y.shape)
    break
for batch in test_dataloader:
    x, y = batch
    print("Test batch shapes:", x.shape, y.shape)
    break
for batch in valid_dataloader:
    x, y = batch
    print("Test batch shapes:", x.shape, y.shape)
    break

#setup training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClass(len(vocab), 128, 256).to(device)
loss_func = torch.nn.BCEWithLogitsLoss()
learning_rate = 1e-3
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
num_epochs = 10
losses = []
val_losses = []
corrects = []
valid_result = []
val_corrects = []
val_loss, val_correct, val_total = 0.0, 0, 0
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=3)
for epoch in range(num_epochs):
    
    for inputs, labels in train_dataloader:
        input = inputs.to(device)
        labels = labels.to(device).unsqueeze(1).float()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
    print(f'epoch {epoch}, loss : {loss.item()}')
    
    model.eval()
    #validation loop
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for inputs_val, labels_val in valid_dataloader:
            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device).unsqueeze(1).float()
            output_val = model(inputs_val)
            loss_val = loss_func(output_val, labels_val)
            val_loss = loss_val
            predicted = torch.sigmoid(output_val) > 0.5
            val_correct += (predicted == labels_val).sum().item()
            val_total += labels_val.size(0)
            val_losses.append(val_loss)
        val_corrects.append(val_correct)
        val_acc = val_correct / val_total
        print(f"Validation [{epoch+1}], val_loss : {val_loss}, val_correct : {val_correct}, Total {val_total}, Accuracy : {val_acc}")
    scheduler.step(val_loss)

    
model.eval()

correct = 0 
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for input_ids, labels in test_dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device).unsqueeze(1).float()
        output = model(input_ids)
        predicted = torch.sigmoid(output) > 0.5
        
        all_preds.append(output.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        correct += (predicted == labels.bool()).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    print(f"Accuracy test {accuracy:.4f}")



