import pandas as pd
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# Tokenizer

# ...existing code...
tokenizer = get_tokenizer("basic_english")

# Load and preprocess train data
df_text = pd.read_csv('D:\MyFiles\Datasets\AmazonReview\amazon_review_polarity_csv\train.csv')
df_text.columns = ["Label", "Subject", "Body"]
df_text_splited = df_text[:150000].copy().reset_index(drop=True)
df_text_splited = df_text_splited.dropna()
label_variations = df_text_splited['Label'].unique()
df_text_splited['Label'] = df_text_splited["Label"].map(lambda x: 0 if x == 1 else 1)
df_text_splited['Subject'] = df_text_splited['Subject'].fillna('')
df_text_splited['Body'] = df_text_splited['Body'].fillna('')

def remove_emojis(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def clean_text(text):
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s.,!?:;\'"()\[\]{}@#$%^&*+=\-/\\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removeHtmlTags(string):
    s2 = re.sub(r"<.*?>", "", string)
    return s2

df_text_splited["Subject"] = df_text_splited["Subject"].apply(remove_emojis)
df_text_splited["Body"] = df_text_splited["Body"].apply(remove_emojis)
df_text_splited["Subject"] = df_text_splited["Subject"].apply(removeHtmlTags)
df_text_splited["Body"] = df_text_splited["Body"].apply(removeHtmlTags)
df_text_splited["Subject"] = df_text_splited["Subject"].apply(clean_text)
df_text_splited["Body"] = df_text_splited["Body"].apply(clean_text)

def generate_token(dt, tokenizer):
    for text in dt:
        yield tokenizer(text)
gen = generate_token(df_text_splited["Subject"]+df_text_splited["Body"], tokenizer)
vocab = build_vocab_from_iterator(gen, specials=["<unk>", "<pad>"], max_tokens=15000)
vocab.set_default_index(vocab["<unk>"])
index_pad = vocab["<pad>"]

def pad_and_encode(data):
    tokens = tokenizer(data)
    indexes = [vocab[token] for token in tokens]
    if len(indexes) < 256:
        padding_length = 256 - len(indexes)
        indexes += [index_pad] * padding_length
    elif len(indexes) > 256:
        indexes = indexes[0:256]
    return indexes

df_text_splited["ConcatString"] = df_text_splited["Subject"] + " " + df_text_splited["Body"]
df_text_splited["input_ids"] = df_text_splited["ConcatString"].apply(pad_and_encode)

class AmazonDataset(Dataset):
    def __init__(self, input_ids_list, labels_list):
        self.inputs = input_ids_list
        self.labels = labels_list
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        return input_tensor, label_tensor

df_text_splited = df_text_splited.reset_index(drop=True)
train_loader_ = AmazonDataset(df_text_splited["input_ids"],df_text_splited["Label"])
train_iter_loader = DataLoader(train_loader_, batch_size = 32, shuffle=True)
for barch in train_iter_loader:
    x, y = barch
    #print(x.shape,y.shape)

class CustomGRU(nn.Module):
    def __init__(self, vocab_size, embeding_dim, hidden_size):
        super().__init__()
        bidirectional = True
        self.embeding = nn.Embedding(vocab_size, embeding_dim)
        self.gru = nn.GRU(input_size = embeding_dim ,hidden_size = hidden_size, bidirectional=True, num_layers=2,batch_first=True,dropout=0.25)
        feat_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(feat_dim)
        self.lin = nn.Linear(feat_dim, 1)
    def forward(self, x):
        x = self.embeding(x)                           # [B, T, E]
        out, h = self.gru(x)                        # out: [B, T, H*D], h: [L*D, B, H]
        h_last = h[-1] if self.gru.bidirectional is False else torch.cat([h[-2], h[-1]], dim=1)
        h_last = self.norm(h_last)                  # [B, feat_dim]
        logits = self.lin(h_last)     
        return logits

# Test data preprocessing
def remove_emojis(text):
    if not isinstance(text, str):
        text = str(text)
    return text.encode("ascii", "ignore").decode("ascii")

def removeHtmlTags(string):
    if not isinstance(string, str):
        string = str(string)
    return re.sub(r"<.*?>", "", string)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s.,!?:;\'"()\[\]{}@#$%^&*+=\-/\\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_text_test = pd.read_csv('D:\MyFiles\Datasets\AmazonReview\amazon_review_polarity_csv\test.csv')
df_text_test.columns = ["Label", "Subject", "Body"]
df_text_test.tail()
df_text_test["Subject"] = df_text_test["Subject"].dropna()
df_text_test["Subject"] = df_text_test["Subject"].apply(remove_emojis)
df_text_test["Body"] = df_text_test["Body"].apply(remove_emojis)
df_text_test["Subject"] = df_text_test["Subject"].apply(clean_text)
df_text_test["Body"] = df_text_test["Body"].apply(clean_text)
df_text_test["ConcatString"] = df_text_test["Subject"] + " " + df_text_test["Body"]
df_text_test["input_ids"] = df_text_test["ConcatString"].apply(pad_and_encode)
df_text_test['Label'] = df_text_test["Label"].map(lambda x: 0 if x == 1 else 1)
df_text_test = df_text_test.reset_index(drop=True)
test_loader_ = AmazonDataset(df_text_test["input_ids"],df_text_test["Label"])
test_loader = DataLoader(test_loader_, batch_size=32, shuffle=False)
for batch in test_loader:
    x, y = batch
    #print(x.shape, y.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomGRU(len(vocab), 128, 256).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 2
losses = []
for epoch in range(num_epochs):
    for inputs, labels in train_iter_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1).float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        optim.zero_grad()
    print(f'epoch {epoch}, loss : {loss.item()}')
    losses.append(loss.item())

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

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(set(all_preds))
print("Уникальные значения в метках:", np.unique(all_labels))
print("Уникальные значения в предсказаниях:", np.unique(all_preds))
df_text_test[df_text_test['Label'] == 0] 
# НУЖНО РЕШИТЬ ПРОБЛЕМУ С МЕТКАМИ СЕЙЧАС [1,2] -> [0,1]
