import pandas as pd
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import json
from sklearn.metrics import confusion_matrix
import spacy
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
from Earlystop import EarlyStopping as es
class PoolingLayer(nn.Module):
    def __init__(self, pad_index):
        super().__init__()
        self.pad_index = pad_index
    def forward(self, x, input_ids, type):
        pad_mask_ = (input_ids != self.pad_index).unsqueeze(-1).float()
        x_masked = x * pad_mask_
        sum_emb = torch.sum(x_masked, dim = 1)
        num_tokens = torch.sum(pad_mask_, dim = 1)
        result = None
        if type == 1:
            result = torch.max(x_masked, dim = 1).values
        else:
            result = sum_emb/num_tokens
        return result

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
        return x

class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embeding_dim, pad_index):
        super().__init__()
        print("EmbDim :", embeding_dim)
        self.emb = nn.Embedding(vocab_size, embeding_dim)
        self.pos_emb = nn.Embedding(512, embeding_dim)
        self.attention = MultiheadAttention(embeding_dim, pad_index, 8)
        self.dropAttention = nn.Dropout(0.25)
        #self.normAttention = nn.LayerNorm(embeding_dim)
        self.attention2 = MultiheadAttention(embeding_dim, pad_index, 4)
        self.attention3 = MultiheadAttention(embeding_dim, pad_index, 2)
        self.pooling = PoolingLayer(pad_index)
        self.norm = nn.LayerNorm(embeding_dim)
        self.drop = nn.Dropout(0.3)
        self.lin = nn.Linear(embeding_dim, 1)
    def forward(self, x):
        self.input_ids = x
        word_emb = self.emb(x)
        positions = torch.arange(0, x.size(1)).to(x.device)
        pos_x = self.pos_emb(positions)

        x = word_emb + pos_x
        x1 = self.attention(x, self.input_ids)
        x1 = self.dropAttention(x1)
        x1 = self.norm(x1)
        x2 = self.attention2(x1, self.input_ids) + x1
        x2 = self.norm(x2)
        x2 = self.dropAttention(x2)
        x3 = self.attention3(x2, self.input_ids) + x2
        x3 = self.norm(x3)
        x3 = self.dropAttention(x3)
        x = self.pooling(x3, self.input_ids, type = 0)
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
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target)



def prepareAmazonDataset():
    tokenizer = get_tokenizer("basic_english")
    def remove_emojis(text):
        return text.encode('ascii', 'ignore').decode('ascii')

    def clean_text(text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def generate_token_spacy(dt, tokenizer):
        for text in dt:
            yield tokenizer(text)
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
    df_text_splited['label'] = df_text_splited["label"].map(lambda x: 0 if x == 1 else 1)
    df_text_splited['Subject'] = df_text_splited['Subject'].fillna('')
    df_text_splited['Body'] = df_text_splited['Body'].fillna('')
    df_text_splited["Subject"] = df_text_splited["Subject"].apply(remove_emojis)
    df_text_splited["Body"] = df_text_splited["Body"].apply(remove_emojis)
    df_text_splited["Subject"] = df_text_splited["Subject"].apply(removeHtmlTags)
    df_text_splited["Body"] = df_text_splited["Body"].apply(removeHtmlTags)
    df_text_splited["Subject"] = df_text_splited["Subject"].apply(clean_text)
    df_text_splited["Body"] = df_text_splited["Body"].apply(clean_text)
    gen = generate_token_spacy(df_text_splited["Subject"]+df_text_splited["Body"], tokenizer)
    vocab = build_vocab_from_iterator(gen, specials=["<unk>", "<pad>"], max_tokens=30000)
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
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'Aspect ratio:.*?(?=\n|$)', '', text)
        text = re.sub(r'Sound format:.*?(?=\n|$)', '', text)
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
    
    max_tokens = 25000
    vocab = build_vocab_from_iterator(gen, specials=['<unk>', '<pad>'], max_tokens=max_tokens)
    vocab.set_default_index(vocab["<unk>"])
    data["input_ids"] = data["review"].apply(pad_and_encode, index_pad=vocab["<pad>"])
    stoi = vocab.get_stoi()
    with open("vocab.json", "w", encoding = "utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    vocab.set_default_index(vocab["<unk>"])
    data["input_ids"] = data["review"].apply(pad_and_encode, index_pad=vocab["<pad>"])
    stoi = vocab.get_stoi()
    with open("vocab.json", "w", encoding = "utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    print("Default index:", vocab.get_default_index())
    print("Index of <pad>:", vocab["<pad>"])
    print("Index test :", vocab["13dfsdafsf"])
    return data, vocab
runTrain = True
dataset = "Amazon" #"IMDB"  #"Amazon"

data, vocab = (prepareDataForImdb() if dataset == "IMDB" else prepareAmazonDataset())

print(data[data["label"] == 0].shape[0], "positive samples")
data['lenStr'] = data['review'].apply(lambda x: len(x))

print(data['lenStr'].describe())
print("Count positive ",data[data["label"] == 1]["lenStr"].describe())
print("Count negative ",data[data["label"] == 0]["lenStr"].describe())
count_input_unk = data[data["input_ids"].apply(lambda x: vocab["<unk>"] in x)].shape[0]
count_input_unk_without = data[data["input_ids"].apply(lambda x: vocab["<unk>"] not in x)].shape[0]
print("Count of samples with <unk> token:", count_input_unk)
print("Count of samples without <unk> token:", count_input_unk_without)
print("Count vocab tok percentage :" ,count_input_unk / data.shape[0] * 100)#


train_, test_ = train_test_split(data, test_size=0.3, random_state=45)
train_, valid_ = train_test_split(train_, test_size=0.2, random_state=32)#
test_ = test_.reset_index(drop = True)#


print("Balance of values in train set ", train_['label'].value_counts())
print("Balance of values in valid set ", valid_['label'].value_counts())
print("Balance of values in test set ", test_['label'].value_counts())

train_ = LabelsIdsDataset(train_["input_ids"].tolist(), train_["label"].tolist())
test_dataloader = LabelsIdsDataset(test_["input_ids"].tolist(), test_["label"].tolist())
valid_ = LabelsIdsDataset(valid_["input_ids"].tolist(), valid_["label"].tolist())#
val_filter_first, val_filter_second = 300, 600 #
train_dataloader = DataLoader(train_, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataloader, batch_size=32, shuffle=False)
valid_dataloader = DataLoader(valid_, batch_size=32, shuffle=True)#

#setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClass(len(vocab), 256, vocab["<pad>"]).to(device)
loss_func = torch.nn.BCEWithLogitsLoss()

data['label'].value_counts()

#learning_rate = 9e-4 most optimal for IMDB
learning_rate = 8e-4
optim = torch.optim.AdamW(model.parameters(), lr = learning_rate)
num_epochs = 12
losses = []
val_losses = []
corrects = []
valid_result = []
val_corrects = []
val_loss, val_correct, val_total = 0.0, 0, 0
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=3, factor=0.5)
early_stopping = es(patience=3, min_delta=0.0001)
val_acc_array = []

print("Last layer bias:", model.lin.bias.item())
print("Last layer weight mean:", model.lin.weight.mean().item())

if(runTrain):
    for epoch in range(num_epochs):#
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
        print(f'epoch {epoch}, loss : {loss.item()}')#
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
                val_losses.append(val_loss.detach().cpu().numpy())
            val_corrects.append(val_correct)
            val_acc = val_correct / val_total
            val_acc_array.append(val_acc)
            print(f"Validation [{epoch}], val_loss : {val_loss}, val_correct : {val_correct}, Total {val_total}, Accuracy : {val_acc}")
            if early_stopping(val_loss, model.state_dict()):
                print("Loading best model weights")
                model.load_state_dict(early_stopping.get_best_weights())
                break
        print(f"Current LR: {optim.param_groups[0]['lr']}")
        scheduler.step(val_loss)#


if(runTrain == False):
    if dataset == "IMDB":
        model.load_state_dict(torch.load("multihead_transformer_IMDB_0.84.pth"))
    else:
        model.load_state_dict(torch.load("multihead_transformer_Amazon_0.87.pth"))
    model.eval()#

fineTuneFlag = False
if(fineTuneFlag and runTrain == False):
    print("Fine-tuning model...")
    model.emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    model.attention.key.weight.requires_grad = False
    model.attention.value.weight.requires_grad = False
    model.attention.query.weight.requires_grad = False
    #model.attention.norm.weight.requires_grad = False
    data, vocab = prepareDataForImdb()
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
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
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

        all_preds.append(predicted.cpu().numpy())
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
all_probs = []
all_pred_ids = []
all_labels = []
incorrect_vals = []
probs = []
obj_for_debug = []
obj_for_right_pred = []
accuracy = 0
labels_probs = []
#if runTrain:
with torch.no_grad():
    for batch_id, (input_ids, labels) in enumerate(test_dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device).unsqueeze(1).float()
        output = model(input_ids)
        predicted = torch.sigmoid(output) > 0.55
        
        all_preds.append(predicted.cpu().numpy())
        all_pred_ids.append(output)
        all_labels.append(labels.cpu().numpy())
        example_start_idx = batch_id * test_dataloader.batch_size
        labels_bool = labels.bool()
        all_probs.append(torch.sigmoid(output).squeeze().cpu().numpy())
        probs_ = torch.sigmoid(output).squeeze().cpu().numpy()
        probs.append(probs_.flatten())
        labels_probs.append(labels.squeeze().cpu().numpy().flatten())

        for i in range(len(predicted)):
            globax_idx = example_start_idx + i
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
                    "input_ids" : input_ids[i].cpu().numpy().tolist(),
                    "uncertain" : True if abs(torch.sigmoid(output[i]) - 0.5) < 0.1 else False
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
print(len(obj_for_debug))#

all_probs_np = np.concatenate(all_probs)
all_labels_np = np.concatenate(all_labels)
all_probs_np_squeezed = np.concatenate(probs)
all_labels_np_squeezed = np.concatenate(labels_probs)

for threshold in [0.5, 0.55, 0.6, 0.65, 0.658]:
    predicted = all_probs_np_squeezed > threshold
    accuracy = (predicted == all_labels_np_squeezed).sum() / len(all_labels_np_squeezed)
    print(f"Threshold {threshold:.3f}: Accuracy {accuracy:.4f}")

### DEBUG#
##
#
print("First batch probs:", all_probs[0])
print("First batch probs shape:", all_probs[0].shape)

print("Shape first element:", all_probs[0].shape if hasattr(all_probs[0], 'shape') else len(all_probs[0]))

all_probs_np = np.concatenate(all_probs)
all_labels_np = np.concatenate(all_labels)
all_labels_np = all_labels_np.squeeze()
fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
# Найди оптимальный порог (где TPR-FPR максимальна)


optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("Treshold for classification:", optimal_threshold)
import matplotlib.pyplot as plt

def show_roc_auc_graph(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
def show_histogream_with_distrib_fp_fn(obj_for_debug):
    false_neg = [i["textLen"] for i in obj_for_debug if i["color_metric"] == "False Negative"]
    false_pos = [i["textLen"] for i in obj_for_debug if i["color_metric"] == "False Positive"]  

    plt.figure(figsize=(12, 5))
    plt.hist([false_neg, false_pos], bins=30, label=['False Negative', 'False Positive'], 
             color=['red', 'blue'], alpha=0.6)
    plt.xlabel('Text Length')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Text Lengths in Errors')
    plt.show()
def show_plot_with_distrib_of_percentage(all_probs_sigmoid, optimal_threshold=0.65):
    plt.figure(figsize=(10, 5))
    plt.hist(all_probs_sigmoid, bins=50, edgecolor='black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Model Predictions')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (0.5)')
    plt.axvline(x=optimal_threshold, color='g', linestyle='--', label='Optimal threshold (0.65)')
    plt.legend()
    plt.show()

all_probs_flat = np.concatenate([p.squeeze().cpu().numpy() for p in all_pred_ids])
all_probs_sigmoid = 1 / (1 + np.exp(-all_probs_flat))  # Если это логиты
# Гистограмма длин текстов для ложных отрицательных и ложных положительных ошибок
show_histogream_with_distrib_fp_fn(obj_for_debug)
show_plot_with_distrib_of_percentage(all_probs_sigmoid,optimal_threshold)
show_roc_auc_graph(fpr, tpr)





false_positives = [e for e in obj_for_debug if e["color_metric"] == "False Positive"]
false_negatives = [e for e in obj_for_debug if e["color_metric"] == "False Negative"]
print(f"False Positives: {len(false_negatives)}")

for i in range(5):
    print(f"\n{i+1}. Текст: {false_negatives[i]['text'][:300]}")
correct = 0
total = 0
accuracy_test = 0
from collections import Counter
tokens_list = []
tokens_list_pos = []


for i in range(2000):
    tokens = obj_for_debug[i]["input_ids"]
    fist_token_pad = tokens.index(vocab["<pad>"]) if vocab["<pad>"] in tokens else len(tokens)
    tokens_without_pad = tokens[0:fist_token_pad]
    tokens_list.extend(tokens_without_pad)#

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
        counter_values.append(i[0])#

import tools.vector_checking as vc
tokens_weight = None
with torch.no_grad():
    tokens_weight = model.emb.weight.data[counter_values].cpu().detach().numpy()
data_for_scatter = []
data_for_scattter_cos = []
print(tokens_weight[0][0])#

coordinates_evc, coordinates_cos = vc.main(tokens_weight)#
result_ngrams = vc.extract_nrgams(tokens_list, n = 3)
result_ngrams_pos = vc.extract_nrgams(tokens_list_pos, n = 3)#



arr_items_neg = []
arr_items_pos = []

counter_Ngrams = Counter(result_ngrams)
counter_PosNrgams = Counter(result_ngrams_pos)
for item in counter_Ngrams.most_common(20):
    arr_items_neg.append({"ngram_text": vocab_itos[item[0][0]]+vocab_itos[item[0][1]]+vocab_itos[item[0][2]], "count": item[1], "color" : "red"})
for item in counter_PosNrgams.most_common(20):
    arr_items_pos.append({"ngram_text": vocab_itos[item[0][0]]+vocab_itos[item[0][1]]+vocab_itos[item[0][2]], "count": item[1], "color" : "green"})#


arr_temp = []
arr_temp.extend(arr_items_neg)
arr_temp.extend(arr_items_pos)#

## print ngram bar plot#

fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(20, 20))
plt.subplots_adjust(wspace=2, hspace=3)
conf_mat = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_preds))
ax1.bar([item["ngram_text"] for item in arr_temp], [item["count"] for item in arr_temp], color=[item["color"] for item in arr_temp], label='Negative Positive N-grams', width=0.6)
ax2.set_title("Confusion Matrix")
ax2.set_xlabel("Predicted Label")
ax2.set_ylabel("True Label")
ax3.plot(range(0, len(val_acc_array)),val_acc_array,label='Validation Accuracy')
sns.heatmap(conf_mat, annot=True, fmt="d", cmap='Blues', ax=ax2)#

precision = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1])
recall = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0])#

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")#

#class Debugger:
#    def __init__(self, vocab, model):
#        self.vocab = vocab
#        self.model = model
#    def build(self):
#        return self#
#

## Debuger builder for transformer model
#class TransformerDebugger(Debugger):
#    def __init__(self, vocab, model):
#        super().__init__(vocab, model)
#    def build_word_emb_metrics(self, tokens_spec, token_words):
#        print("Debug info word embeddings.....")
#        #tokens_spec = ["<unk>", '.', '<pad>', '\'', ',', '!']
#        #token_words =['of', 'movie', 'good', 'great', 'like', 'story']
#        weights_spec = []
#        weights_word = []
#        with torch.no_grad():
#            for i in tokens_spec:
#                weights_spec.append(self.model.emb.weight[vocab[i]].data.cpu().detach().numpy())
#            for i in token_words:
#                weights_word.append(self.model.emb.weight[vocab[i]].data.cpu().detach().numpy())
#        arr_cos_words = []
#        arr_cos_spec = []#

#        for i in range(len(weights_word)):
#            result = cosine(weights_spec[0], weights_word[i])
#            print("Cosine between <unk> and", token_words[i], ":", result)
#            arr_cos_words.append(cosine(weights_spec[0], weights_word[i]))
#        for i in range(1, len(weights_spec)):
#            result = cosine(weights_spec[0], weights_spec[i])
#            print("Cosine between <unk> and", tokens_spec[i], ":", result)
#            arr_cos_spec.append(cosine(weights_spec[0], weights_spec[i]))#

#        print("Special tokens weights:", cosine(weights_spec[0], weights_word[0]))
#        print("Special tokens weights:", cosine(weights_spec[0], weights_word[1]))#

#        print("Norm of spec tok", torch.norm(torch.tensor(weights_spec[0])))
#        print("Norm of of word", torch.norm(torch.tensor(weights_word[0])))#

#        return self
#    def build_conf_matrix(self, all_labels, all_preds):
#        self.conf_matrix = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_preds))
#        return self
#    def build_precision_recall(self):
#        self.precision = self.conf_matrix[1][1] / (self.conf_matrix[1][1] + self.conf_matrix[0][1])
#        self.recall = self.conf_matrix[1][1] / (self.conf_matrix[1][1] + self.conf_matrix[1][0])
#        return self
#    def build_ngrams_analysis(self, tokens_list, tokens_list_pos, n = 3):
#        from collections import Counter
#        from scipy.spatial.distance import cosine
#        import tools.vector_checking as vc
#        result_ngrams = vc.extract_nrgams(tokens_list, n = n)
#        result_ngrams_pos = vc.extract_nrgams(tokens_list_pos, n = n)
#        self.counter_Ngrams = Counter(result_ngrams)
#        self.counter_PosNrgams = Counter(result_ngrams_pos)
#        return self#

def debug_info_word_emb(vocab, model):#
    print("Debug info word embeddings.....")
    tokens_spec = ["<unk>", '.', '<pad>', '\'', ',', '!']
    token_words =['of', 'movie', 'good', 'great', 'like', 'story']
    weights_spec = []
    weights_word = []
    with torch.no_grad():
        for i in tokens_spec:
            weights_spec.append(model.emb.weight[vocab[i]].data.cpu().detach().numpy())
        for i in token_words:
            weights_word.append(model.emb.weight[vocab[i]].data.cpu().detach().numpy())
    arr_cos_words = []
    arr_cos_spec = []#
    for i in range(len(weights_word)):
        result = cosine(weights_spec[0], weights_word[i])
        print("Cosine between <unk> and", token_words[i], ":", result)
        arr_cos_words.append(cosine(weights_spec[0], weights_word[i]))
    for i in range(1, len(weights_spec)):
        result = cosine(weights_spec[0], weights_spec[i])
        print("Cosine between <unk> and", tokens_spec[i], ":", result)
        arr_cos_spec.append(cosine(weights_spec[0], weights_spec[i]))
    
    print("Special tokens weights:", cosine(weights_spec[0], weights_word[0]))
    print("Special tokens weights:", cosine(weights_spec[0], weights_word[1]))#
    print("Norm of spec tok", torch.norm(torch.tensor(weights_spec[0])))
    print("Norm of of word", torch.norm(torch.tensor(weights_word[0])))


debug_info_word_emb(vocab, model)
uncertrain = [i for i in obj_for_debug if i["uncertrain"] == True]

print("Uncertrain cases:")
for i in uncertrain:
    print(i)


tokens_spec = ["<unk>", '.', '<pad>', '\'', ',', '!']
token_words =['of', 'movie', 'good', 'great', 'like', 'story']
weights_spec = []
weights_word = []


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


if(runTrain):
    torch.save(model.state_dict(), f"multihead_transformer_{dataset}_{accuracy:.2f}.pth")
plt.show()