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
        self.projection = nn.Linear(size_kernel, size_kernel)
    def forward(self, x, input_ids):
        residual_x = x
        x_k = self.key(x)
        x_v = self.value(x)
        x_q = self.query(x)
        x_q_head = x_q.view(x_q.shape[0], x_q.shape[1], self.num_heads, self.size // self.num_heads)
        x_k_head = x_k.view(x_k.shape[0], x_k.shape[1], self.num_heads, self.size // self.num_heads)
        x_v_head = x_v.view(x_v.shape[0], x_v.shape[1], self.num_heads, self.size // self.num_heads)
        pad_mask = (input_ids != self.pad_index)
        results = []
        mask_rows = pad_mask.unsqueeze(-2)
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
            #print(f"Голова {i}: result_mat.shape = {result_mat.shape}")
        
            results.append(result_mat)
        result = torch.cat(results, dim = -1)
        # residual  connection
        result = self.projection(result)
        x = result + residual_x
        
        x = self.norm(x)
        x = self.drop(x)
        pad_mask_ = (input_ids != self.pad_index).unsqueeze(-1).float()
        x_masked = x * pad_mask_
        sum_emb = torch.sum(x_masked, dim = 1)
        num_tokens = torch.sum(pad_mask_, dim = 1)

        num_tokens = torch.clamp(num_tokens, min = 1)
        mean_ = sum_emb/num_tokens
        return mean_


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
        residual_x = x
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




def prepareAmazonDataset():
    tokenizer = get_tokenizer("basic_english")
    def remove_emojis(text):
        return text.encode('ascii', 'ignore').decode('ascii')

    def clean_text(text):
        text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s.,!?:;\'"()\[\]{}@#$%^&*+=\-/\\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def generate_token(dt, tokenizer):
        for text in dt:
            yield tokenizer(text)
    def removeHtmlTags(string):
        s2 = re.sub(r"<.*?>", "", string)
        return s2
    def pad_and_encode(data):
        tokens = tokenizer(data)
        indexes = [vocab[token] for token in tokens]
        if len(indexes) < 256:
            padding_length = 256 - len(indexes)
            indexes += [index_pad] * padding_length
        elif len(indexes) > 256:
            indexes = indexes[0:256]
        return indexes
    # Load and preprocess train data
    df_text = pd.read_csv('D:\\MyFiles\\Datasets\\AmazonReview\\amazon_review_polarity_csv\\train.csv')
    df_text.columns = ["label", "Subject", "Body"]
    df_text_splited = df_text[:90000].copy().reset_index(drop=True)
    df_text_splited = df_text_splited.dropna()
    label_variations = df_text_splited['label'].unique()
    df_text_splited['label'] = df_text_splited["label"].map(lambda x: 0 if x == 1 else 1)
    df_text_splited['Subject'] = df_text_splited['Subject'].fillna('')
    df_text_splited['Body'] = df_text_splited['Body'].fillna('')
    df_text_splited["Subject"] = df_text_splited["Subject"].apply(remove_emojis)
    df_text_splited["Body"] = df_text_splited["Body"].apply(remove_emojis)
    df_text_splited["Subject"] = df_text_splited["Subject"].apply(removeHtmlTags)
    df_text_splited["Body"] = df_text_splited["Body"].apply(removeHtmlTags)
    df_text_splited["Subject"] = df_text_splited["Subject"].apply(clean_text)
    df_text_splited["Body"] = df_text_splited["Body"].apply(clean_text)
    gen = generate_token(df_text_splited["Subject"]+df_text_splited["Body"], tokenizer)
    vocab = build_vocab_from_iterator(gen, specials=["<unk>", "<pad>"], max_tokens=17000)
    vocab.set_default_index(vocab["<unk>"])
    index_pad = vocab["<pad>"]
    df_text_splited["review"] = df_text_splited["Subject"] + " " + df_text_splited["Body"]
    df_text_splited["input_ids"] = df_text_splited["review"].apply(pad_and_encode)
    df_text_splited = df_text_splited.reset_index(drop=True)
    return df_text_splited, vocab

def prepareDataForImdb():


    tokenizer = get_tokenizer("basic_english")
    def labelFix(dt):
        if dt == 'positive':
            return 1
        else:
            return 0
    def gen_tokenizer(data, tokenizer):
        for text in data:
            yield tokenizer(text)
    def removeHtmlTags(string):
        s2 = re.sub(r"<.*?>", "", string)
        return s2
    def pad_and_encode(data, index_pad):
        tokens = tokenizer(data)
        indexes = [vocab[token] for token in tokens]
        if len(indexes) < 256:
            padding_length = 256 - len(indexes)
            indexes += [index_pad] * padding_length
        elif len(indexes) > 256:
            indexes = indexes[0:256]
        return indexes
    def clean_text(text):
        text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s.,!?:;\'"()\[\]{}@#$%^&*+=\-/\\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    data = pd.read_csv('D:\MyFiles\Datasets\IMDB\IMDB Dataset.csv')
    data.columns = ['review', 'label']
    data['label'] = data['label'].apply(labelFix)
    data['review'] = data['review'].apply(removeHtmlTags)
    data['review'] = data['review'].apply(clean_text)
    def removeHtmlTags(string):
        s2 = re.sub(r"<.*?>", "", string)
        return s2
    
    data['review'] = data['review'].apply(removeHtmlTags)
    
    def gen_tokenizer(data, tokenizer):
        for text in data:
            yield tokenizer(text)
    
    gen = gen_tokenizer(data['review'], tokenizer=tokenizer)
    #max_tokens = (10000 if dataset == "IMDB" else 15000)
    max_tokens = 17000
    vocab = build_vocab_from_iterator(gen, specials=['<unk>', '<pad>'], max_tokens=max_tokens)
    vocab.set_default_index(vocab["<unk>"])
    data["input_ids"] = data["review"].apply(pad_and_encode, index_pad=vocab["<pad>"])
    stoi = vocab.get_stoi()
    with open("vocab.json", "w", encoding = "utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    gen = gen_tokenizer(data['review'], tokenizer=tokenizer)
    #vocab = build_vocab_from_iterator(gen, specials=['<unk>', '<pad>'], max_tokens=15000)
    #vocab.set_default_index(vocab["<unk>"])
    data["input_ids"] = data["review"].apply(pad_and_encode, index_pad=vocab["<pad>"])
    stoi = vocab.get_stoi()
    with open("vocab.json", "w", encoding = "utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    print("Default index:", vocab.get_default_index())
    print("Index of <pad>:", vocab["<pad>"])
    print("Index test :", vocab["13dfsdafsf"])
    return data, vocab
runTrain = False
dataset = "IMDB" #"IMDB"  #"Amazon"

data, vocab = (prepareDataForImdb() if dataset == "IMDB" else prepareAmazonDataset())
count_input_unk = data[data["input_ids"].apply(lambda x: vocab["<unk>"] in x)].shape[0]
print("Count of samples with <unk> token:", count_input_unk)
print("Count vocab tok percentage :" ,count_input_unk / data.shape[0] * 100)


train_, test_ = train_test_split(data, test_size=0.3, random_state=45)
train_, valid_ = train_test_split(train_, test_size=0.2, random_state=32)

test_ = test_.reset_index(drop = True)


train_ = LabelsIdsDataset(train_["input_ids"].tolist(), train_["label"].tolist())
test_dataloader = LabelsIdsDataset(test_["input_ids"].tolist(), test_["label"].tolist())
valid_ = LabelsIdsDataset(valid_["input_ids"].tolist(), valid_["label"].tolist())

val_filter_first, val_filter_second = 300, 600 

train_dataloader = DataLoader(train_, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataloader, batch_size=32, shuffle=False)
valid_dataloader = DataLoader(valid_, batch_size=32, shuffle=True)



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
model = TransformerClass(len(vocab), 128, vocab["<pad>"]).to(device)
loss_func = torch.nn.BCEWithLogitsLoss()
learning_rate = 2e-4
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
num_epochs = 10
losses = []
val_losses = []
corrects = []
valid_result = []
val_corrects = []
val_loss, val_correct, val_total = 0.0, 0, 0
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=3)

if(runTrain):
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

    
if(runTrain == False):
    if dataset == "IMDB":
        model.load_state_dict(torch.load("multihead_transformer_IMDB_0.84.pth"))
    else:
        model.load_state_dict(torch.load("multihead_transformer_Amazon_0.87.pth"))
    model.eval()

fineTuneFlag = True
if(fineTuneFlag and runTrain == False):
    print("Fine-tuning model...")
    model.emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    model.attention.key.weight.requires_grad = False
    model.attention.value.weight.requires_grad = False
    model.attention.query.weight.requires_grad = False
    #model.attention.norm.weight.requires_grad = False
    data, vocab = prepareAmazonDataset()
    train_, test_ = train_test_split(data, test_size=0.3, random_state=45)
    train_, valid_ = train_test_split(train_, test_size=0.2, random_state=32)
    test_ = test_.reset_index(drop = True)
    train_dataset = LabelsIdsDataset(train_["input_ids"].tolist(), train_["label"].tolist())
    valid_dataset = LabelsIdsDataset(valid_["input_ids"].tolist(), valid_["label"].tolist())
    test_dataset = LabelsIdsDataset(test_["input_ids"].tolist(), test_["label"].tolist())
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    learning_rate = 1e-4
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
    correct = 0 
    total = 0
    all_preds = []
    accuracy = 0
    all_pred_ids = []
    all_labels = []
    incorrect_vals = []
    obj_for_debug = []
    obj_for_right_pred = []
    num_epochs = 3
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
        #scheduler.step(val_loss)
    #with torch.no_grad():
    for batch_id, (input_ids, labels) in enumerate(test_dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device).unsqueeze(1).float()
        output = model(input_ids)
        predicted = torch.sigmoid(output) > 0.5

        all_preds.append(output.detach().cpu().numpy())
        all_pred_ids.append(output)
        all_labels.append(labels.detach().cpu().numpy())
        example_start_idx = batch_id * test_dataloader.batch_size
        labels_bool = labels.bool()
        #for i in range(len(predicted)):
        for i in range(len(predicted)):
            globax_idx = example_start_idx + i
            #obj_for_debug['indexes'].append(globax_idx)
            if predicted[i] != labels_bool[i]:
                text_Res = None
                text = test_["review"][globax_idx]
                if predicted[i] == True and labels_bool[i] == False :
                  text_Res = "False Positive"
                else :
                  text_Res = "False Negative"
                error_info = {
                    "text" : text,
                    "color_metric" : text_Res,
                    "globalIdx" : globax_idx,
                    "textLen" : len(text),
                    "input_ids" : input_ids[i].cpu().numpy().tolist()
                }
                obj_for_debug.append(error_info)
            else:
                correct_info = {
                    "globalIdx" : globax_idx,
                    "input_ids" : input_ids[i].cpu().numpy().tolist()
                }
                obj_for_right_pred.append(correct_info)#
        correct += (predicted == labels.bool()).sum().item()
        total += labels.size(0)
        accuracy = correct / total
    print(f"Accuracy test {accuracy:.4f}")
    print("Errors" , [error["text"] for error in obj_for_debug[0:5]])

    model.save_state_dict(model.state_dict(), f"multihead_transformer_finetuned_Amazon_{accuracy:.2f}.pth")
correct = 0 
total = 0
all_preds = []
all_pred_ids = []
all_labels = []
incorrect_vals = []
obj_for_debug = []
obj_for_right_pred = []
accuracy = 0
#if runTrain:
with torch.no_grad():
    for batch_id, (input_ids, labels) in enumerate(test_dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device).unsqueeze(1).float()
        output = model(input_ids)
        predicted = torch.sigmoid(output) > 0.5
        
        all_preds.append(output.cpu().numpy())
        all_pred_ids.append(output)
        all_labels.append(labels.cpu().numpy())
        example_start_idx = batch_id * test_dataloader.batch_size
        labels_bool = labels.bool()
        #for i in range(len(predicted)):
        for i in range(len(predicted)):
            globax_idx = example_start_idx + i
            #obj_for_debug['indexes'].append(globax_idx)
            if predicted[i] != labels_bool[i]:
                text_Res = None
                text = test_["review"][globax_idx]
                if predicted[i] == True and labels_bool[i] == False :
                  text_Res = "False Positive"
                else :
                  text_Res = "False Negative"
                error_info = {
                    "text" : text,
                    "color_metric" : text_Res,
                    "globalIdx" : globax_idx,
                    "textLen" : len(text),
                    "input_ids" : input_ids[i].cpu().numpy().tolist()
                }
                obj_for_debug.append(error_info)
            else:
                correct_info = {
                    "globalIdx" : globax_idx,
                    "input_ids" : input_ids[i].cpu().numpy().tolist()
                }
                obj_for_right_pred.append(correct_info)

        correct += (predicted == labels.bool()).sum().item()
        total += labels.size(0)
        accuracy = correct / total
    print(f"Accuracy test {accuracy:.4f}")
print(len(obj_for_debug))

## DEBUG



correct = 0
total = 0
accuracy_test = 0

from collections import Counter
tokens_list = []
tokens_list_pos = []

def analyze_token_results(token_id, obj_for_debug):
    fp_count = 0
    fn_count = 0
    for i in obj_for_debug:
        if token_id in i["input_ids"]:
            if i["color_metric"] == "False Positive":
                fp_count += 1
            elif i["color_metric"] == "False Negative":
                fn_count += 1
    if fp_count > fn_count:
        return {"fp_count": fp_count,"fn_count" :fn_count, "dom": "False Positive"}
    elif fn_count > fp_count:
        return {"fp_count": fp_count,"fn_count" :fn_count, "dom": "False Negative"}
    else :
        return {"fp_count": fp_count,"fn_count" :fn_count, "dom": "Balanced"}

for i in range(2000):
    tokens = obj_for_debug[i]["input_ids"]
    fist_token_pad = tokens.index(vocab["<pad>"]) if vocab["<pad>"] in tokens else len(tokens)
    tokens_without_pad = tokens[0:fist_token_pad]
    tokens_list.extend(tokens_without_pad)

for i in range(2000):
    tokens = obj_for_right_pred[i]["input_ids"]
    fist_token_pad = tokens.index(vocab["<pad>"]) if vocab["<pad>"] in tokens else len(tokens)
    tokens_without_pad = tokens[0:fist_token_pad]
    tokens_list_pos.extend(tokens_without_pad)
frequent_tokens = []
counter_values = []
counter = Counter(tokens_list)
vocab_itos = vocab.get_itos()
for i in counter.items():
    if i[1] > 20:
        text = vocab_itos[i[0]]
    
        if re.search(r'[.,!?:;\'"()\[\]{}@#$%^&*+=\-/\\]', text):
            continue
        debug_obj = {
            "token_id" : i[0],
            "count" : i[1],
            "token_str" : text
        }
        frequent_tokens.append(debug_obj)
        counter_values.append(i[0])

import tools.vector_checking as vc
tokens_weight = None
with torch.no_grad():
    tokens_weight = model.emb.weight.data[counter_values].cpu().detach().numpy()
data_for_scatter = []
data_for_scattter_cos = []
print(tokens_weight[0][0])

coordinates_evc, coordinates_cos = vc.main(tokens_weight)

result_ngrams = vc.extract_nrgams(tokens_list, n = 3)
result_ngrams_pos = vc.extract_nrgams(tokens_list_pos, n = 3)

import matplotlib.pyplot as plt



#print("Result bi-grams negative", result_ngrams)

print("Vocab", vocab_itos[2]) 
arr_items_neg = []
arr_items_pos = []
def collect_text_from_ngram(ngram, n):
    arr_temp = []
    for item in counter_Ngrams.most_common(20):
        print("Ngram:", item)
        if n == 2:
            print("Negative ngram tokens", vocab_itos[item[0][0]], vocab_itos[item[0][1]])
        elif n == 3:
            print("Negative ngram tokens", vocab_itos[item[0][0]], vocab_itos[item[0][1]], vocab_itos[item[0][2]])
        arr_temp.append(item)
    return arr_temp
counter_Ngrams = Counter(result_ngrams)
counter_PosNrgams = Counter(result_ngrams_pos)
for item in counter_Ngrams.most_common(20):
    arr_items_neg.append({"ngram_text": vocab_itos[item[0][0]]+vocab_itos[item[0][1]]+vocab_itos[item[0][2]], "count": item[1], "color" : "red"})
for item in counter_PosNrgams.most_common(20):
    arr_items_pos.append({"ngram_text": vocab_itos[item[0][0]]+vocab_itos[item[0][1]]+vocab_itos[item[0][2]], "count": item[1], "color" : "green"})


arr_temp = []
arr_temp.extend(arr_items_neg)
arr_temp.extend(arr_items_pos)

# print ngram bar plot

fig, ax = plt.subplots(figsize=(20, 15))
plt.subplots_adjust(wspace=2, hspace=3)

ax.bar([item["ngram_text"] for item in arr_temp], [item["count"] for item in arr_temp], color=[item["color"] for item in arr_temp], label='Negative Positive N-grams', width=0.6)

plt.show()

tokens_spec = ["<unk>", '.', '<pad>', '\'', ',', '!']
token_words =['of', 'movie', 'good', 'great', 'like', 'story']
weights_spec = []
weights_word = []
import numpy as np
from scipy.spatial.distance import cosine
with torch.no_grad():
    for i in tokens_spec:
        weights_spec.append(model.emb.weight[vocab[i]].data.cpu().detach().numpy())
    for i in token_words:
        weights_word.append(model.emb.weight[vocab[i]].data.cpu().detach().numpy())
arr_cos_words = []
arr_cos_spec = []
for i in range(len(weights_word)):
    result = cosine(weights_spec[0], weights_word[i])
    print("Cosine between <unk> and", token_words[i], ":", result)
    arr_cos_words.append(cosine(weights_spec[0], weights_word[i]))
for i in range(1, len(weights_spec)):
    result = cosine(weights_spec[0], weights_spec[i])
    print("Cosine between <unk> and", tokens_spec[i], ":", result)
    arr_cos_spec.append(cosine(weights_spec[0], weights_spec[i]))

print("Special tokens weights:", cosine(weights_spec[0], weights_word[0]))
print("Special tokens weights:", cosine(weights_spec[0], weights_word[1]))

print("Norm of spec tok", torch.norm(torch.tensor(weights_spec[0])))
print("Norm of of word", torch.norm(torch.tensor(weights_word[0])))

if(runTrain):
    torch.save(model.state_dict(), f"multihead_transformer_{dataset}_{accuracy:.2f}.pth")
