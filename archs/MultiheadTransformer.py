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


class MultiheadAttention(nn.Module):
    def __init__(self, size_kernel, pad_index, num_heads):
        super().__init__()
        self.size = size_kernel
        self.pad_index = pad_index
        self.num_heads = num_heads
        self.key = nn.Linear(size_kernel, int(size_kernel))
        self.value = nn.Linear(size_kernel, int(size_kernel))
        self.query = nn.Linear(size_kernel, int(size_kernel))
        self.norm = nn.LayerNorm(size_kernel)
        self.drop = nn.Dropout(0.2)
        self.projection = nn.Linear(128, size_kernel)
    def forward(self, x, input_ids):
        residual_x = x
        x_k = self.key(x)
        x_v = self.value(x)
        x_q = self.query(x)
        x_q_head = x_q.view(x_q.shape[0], x_q.shape[1], self.num_heads, self.size // self.num_heads)
        x_k_head = x_k.view(x_k.shape[0], x_k.shape[1], self.num_heads, self.size // self.num_heads)
        x_v_head = x_v.view(x_v.shape[0], x_v.shape[1], self.num_heads, self.size // self.num_heads)
        pad_mask = (input_ids != self.pad_index)
        mask_rows = pad_mask.unsqueeze(-2)
        results = []
        for i in range(self.num_heads):
            x_q_head_val = x_q_head[:,:,i,:]
            x_k_head_val = x_k_head[:,:,i,:]
            x_v_head_val = x_v_head[:,:,i,:]

            transpose_k_head = torch.transpose(x_k_head_val, -2, -1)
            att_score = torch.matmul(x_q_head_val, transpose_k_head)

            score = att_score/math.sqrt(self.size // self.num_heads)
            score = score.masked_fill(~mask_rows, -float('inf'))
            weight = torch.softmax(score, dim=-1)
            result_mat = torch.matmul(weight, x_v_head_val)
        
            results.append(result_mat)
        result = torch.cat(results, dim = -1)
        x = self.projection(result)
        print(f"Норма residual: {torch.norm(residual_x).item():.4f}")
        print(f"Норма attention: {torch.norm(result).item():.4f}")
        print(f"Соотношение: {torch.norm(result).item() / torch.norm(residual_x).item():.4f}")
        # residual  connection
        x = x + residual_x
        x = self.norm(x)
        x = self.drop(x)
                
        #print("After projection", x.shape)
        mean = torch.mean(x, dim =1)
        #print("result after mean", mean)
        return mean


class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embeding_dim, pad_index):
        super().__init__()
        print("EmbDim :", embeding_dim)
        self.emb = nn.Embedding(vocab_size, embeding_dim)
        self.pos_emb = nn.Embedding(256, embeding_dim)
        self.attention = MultiheadAttention(embeding_dim, pad_index, 8)
        self.norm = nn.LayerNorm(embeding_dim)
        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(embeding_dim, 1)
    def forward(self, x):
        self.input_ids = x
        word_emb = self.emb(x)
        positions = torch.arange(0, x.size(1)).to(x.device)
        pos_x = self.pos_emb(positions)

        x = word_emb + pos_x
        x = self.attention(x, self.input_ids)
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
print(vocab["<pad>"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClass(len(vocab), 128, vocab["<pad>"]).to(device)
loss_func = torch.nn.BCEWithLogitsLoss()
learning_rate = 1e-4
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
        inputs = inputs.to(device)
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
    #print(f"Learining rate {scheduler.lr}")

    
model.eval()

correct = 0 
total = 0
all_preds = []
all_labels = []
accuracy = 0
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

torch.save(model.state_dict(), f"transformer{accuracy:.2f}.pth")



