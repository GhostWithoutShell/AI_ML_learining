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
#from torch_geometric.nn import SAGPooling
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
import os
from tokenizers.pre_tokenizers import Whitespace
from transformers import AdamW, get_linear_schedule_with_warmup

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
class FFLayer(nn.Module):
    def __init__(self, embeding, expansion=4):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embeding, embeding*expansion),
            nn.GELU(),
            nn.Linear(embeding*expansion, embeding),
            nn.Dropout(0.2),
        )
    def forward(self, x):
        return self.ff(x)




class MultiheadAttention(nn.Module):
    def __init__(self, size_kernel, pad_index, num_heads):
        super().__init__()
        self.emb_dim = size_kernel
        self.head_dim = size_kernel // num_heads
        self.pad_index = pad_index
        self.num_heads = num_heads
        self.key = nn.Linear(size_kernel, int(size_kernel))
        self.value = nn.Linear(size_kernel, int(size_kernel))
        self.query = nn.Linear(size_kernel, int(size_kernel))
        self.norm = nn.LayerNorm(size_kernel)
        self.drop = nn.Dropout(0.2)
        self.projection = nn.Linear(size_kernel, size_kernel)
        
    def forward(self, x, input_ids):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1,2)
        v = v.transpose(1,2)
        q = q.transpose(1,2)

        att_scores = torch.matmul(q, k.transpose(-2, -1))
        att_scores = att_scores/math.sqrt(self.num_heads)
        mask = (input_ids != self.pad_index).unsqueeze(1).unsqueeze(2)

        att_scores = att_scores.masked_fill(mask == 0, -1e9)
        att_scores = torch.softmax(att_scores, dim=-1)
        self.last_scores = att_scores.detach().cpu()

        result = torch.matmul(att_scores, v)
        result = result.transpose(1, 2).contiguous()
        result = result.view(batch_size, seq_len, self.emb_dim)
        output = self.projection(result)
        return output#, att_scores

class TransformerBlock(nn.Module):
    def __init__(self, size_kernel, pad_index, num_heads):
        super().__init__()
        self.attention = MultiheadAttention(size_kernel, pad_index, num_heads)
        self.norm1 = nn.LayerNorm(size_kernel)
        self.norm2 = nn.LayerNorm(size_kernel)
        self.ff = FFLayer(size_kernel)
        self.drop = nn.Dropout(0.2)
    def forward(self, x, input_ids):
        x = self.norm1(x)
        att_out = self.attention(x, input_ids)
        x = x + self.drop(att_out)
        x = self.norm2(x)
        res_x = x
        x = self.ff(x) + res_x
        x = self.drop(x)
        return x#, scores
    
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, x, mask=None):
        # x: [Batch, Seq, Dim]
        # scores: [Batch, Seq, 1]
        scores = self.attention(x)
        
        if mask is not None:
            # mask: [Batch, Seq] (True там, где паддинг)
            scores = scores.masked_fill(mask.unsqueeze(-1), -1e9)
        
        weights = torch.softmax(scores, dim=1)
        # Сумма векторов, взвешенная их важностью
        return torch.sum(x * weights, dim=1)
class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embeding_dim, pad_index):
        super().__init__()
        self.pad_index = pad_index
        self.emb = nn.Embedding(vocab_size, embeding_dim)
        self.pos_emb = nn.Embedding(embeding_dim, embeding_dim)
        self.attBlock1 = TransformerBlock(embeding_dim, pad_index, 8)
        self.attBlock2 = TransformerBlock(embeding_dim, pad_index, 8)

        self.norm = nn.LayerNorm(embeding_dim)
        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(embeding_dim, 1)
        
    def forward(self, x):
        res_x = x
        self.input_ids = x
        word_emb = self.emb(x)
        positions = torch.arange(0, x.size(1)).to(x.device)
        pos_x = self.pos_emb(positions) 
        mask = (self.input_ids != self.pad_index).unsqueeze(1).unsqueeze(2)
        x = word_emb + pos_x
        x = self.attBlock1(x, self.input_ids)
        x = self.attBlock2(x, self.input_ids)
        
        cls_value = x[:,0,:]
        x = self.norm(cls_value)
        
        
        
        x = self.drop(x)
        x = self.lin(x)
        return x #(scores1, scores2)
        

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
    #tokenizer = get_tokenizer("basic_english")
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
    
    def get_training_corpus():
        for i in range(0, len(df_text_splited), 1000):
            yield df_text_splited['review'][i:i+1000].tolist()
    
    
    # Функция для кодирования с проверкой
    def pad_and_encode(text):
        encoding = tokenizer.encode(text)
        tokens = encoding.ids

        # Проверяем индексы
        max_valid_index = len(vocab.stoi) - 1
        tokens = [min(token, max_valid_index) for token in tokens]

        # Обрезаем или дополняем до длины 256
        if len(tokens) < 256:
            padding_length = 256 - len(tokens)
            tokens = tokens + [vocab.stoi["<pad>"]] * padding_length
            return tokens
        elif len(tokens) > 256:
            # Берем первые 128 токенов
            tokens_first_part = tokens[:128]
            # Берем последние 128 токенов
            tokens_second_part = tokens[-128:]
            # Объединяем
            tokens_first_part.extend(tokens_second_part)
            return tokens_first_part
        else:
        # Если длина точно 256, возвращаем как есть
            return tokens
    
    
    
    # ПРОВЕРКА ДАННЫХ
    print("=== DATA VALIDATION ===")
    #print(f"Vocab size: {len(vocab)}")
    
    
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
    #gen = generate_token_spacy(df_text_splited["Subject"]+df_text_splited["Body"], tokenizer)
    #vocab = build_vocab_from_iterator(gen, specials=["<unk>", "<pad>"], max_tokens=27000)
    #vocab.set_default_index(vocab["<unk>"])
    
    df_text_splited["review"] = df_text_splited["Subject"] + " " + df_text_splited["Body"]
    vocab_size = 35000
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<cls>"])
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>", cls_token="<cls>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    
    # Получаем реальный vocab из токенизатора
    vocab_dict = tokenizer.get_vocab()
    print(f"Real vocab size from tokenizer: {len(vocab_dict)}")
    
    # Создаем mapping для индексов
    class CustomVocab:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.stoi = tokenizer.get_vocab()
            self.itos = {v: k for k, v in self.stoi.items()}
            self.specials = {"<unk>", "<pad>", "<cls>"}
            
        def __getitem__(self, token):
            return self.stoi.get(token, self.stoi["<pad>"])
            
        def __len__(self):
            return len(self.stoi)
            
        def get_stoi(self):
            return self.stoi
            
        def get_itos(self):
            return self.itos
    
    
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    vocab = CustomVocab(tokenizer)
    index_pad = vocab["<pad>"]
    df_text_splited["input_ids"] = df_text_splited["review"].apply(pad_and_encode)
    all_input_ids = [idx for sublist in df_text_splited["input_ids"] for idx in sublist]
    max_index = max(all_input_ids) if all_input_ids else 0
    min_index = min(all_input_ids) if all_input_ids else 0
    #df_text_splited["input_ids"] = df_text_splited["review"].apply(pad_and_encode)
    df_text_splited = df_text_splited.reset_index(drop=True)
    return df_text_splited, vocab
def prepareDataForImdbWordPeace():
    
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    def labelFix(dt):
        if dt == 'positive':
            return 1
        else:
            return 0

    def remove_emojis(text):
        """Удаляет эмодзи и специальные символы Unicode"""
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # эмоции
                               u"\U0001F300-\U0001F5FF"  # символы & пиктограммы
                               u"\U0001F680-\U0001F6FF"  # транспорт & карты
                               u"\U0001F1E0-\U0001F1FF"  # флаги
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_urls(text):
        """Удаляет URL адреса"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_mentions_hashtags(text):
        """Удаляет упоминания и хэштеги"""
        cleaned = re.sub(r'[@#]\w+', '', text)
        return cleaned

    def removeHtmlTags(string):
        s2 = re.sub(r"<.*?>", "", string)
        return s2
        

    # Загрузка и предобработка данных
    data = pd.read_csv('D:/MyFiles/Datasets/IMDB/IMDB Dataset.csv')
    data.columns = ['review', 'label']
    data['label'] = data['label'].apply(labelFix)
    data['review'] = data['review'].apply(removeHtmlTags)
    
    
    # Очистка текста
    def comprehensive_text_cleaner(text):
        if not isinstance(text, str):
            return ""
            
        cleaned_text = text
        cleaned_text = remove_urls(cleaned_text)
        cleaned_text = remove_mentions_hashtags(cleaned_text)
        cleaned_text = remove_emojis(cleaned_text)
        cleaned_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s\.\,\!\?\-\:\(\)\"]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    data['review'] = data['review'].apply(comprehensive_text_cleaner)
    
    # Обучение WordPiece токенизатора
    def get_training_corpus():
        for i in range(0, len(data), 1000):
            yield data['review'][i:i+1000].tolist()
    vocab_size = 32000
    # Настройка тренера для WordPiece
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<cls>"]
    )
    
    # Обучение токенизатора
    
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<cls>"])
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Получаем реальный vocab из токенизатора
    vocab_dict = tokenizer.get_vocab()
    print(f"Real vocab size from tokenizer: {len(vocab_dict)}")
    
    # Создаем mapping для индексов
    class CustomVocab:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.stoi = tokenizer.get_vocab()
            self.itos = {v: k for k, v in self.stoi.items()}
            self.specials = {"<unk>", "<pad>", "<cls>"}
            
        def __getitem__(self, token):
            return self.stoi.get(token, self.stoi["<pad>"])
            
        def __len__(self):
            return len(self.stoi)
            
        def get_stoi(self):
            return self.stoi
            
        def get_itos(self):
            return self.itos
    
    vocab = CustomVocab(tokenizer)
    
    # Функция для кодирования с проверкой
    def pad_and_encode(text):
        max_len = 256
        encoding = tokenizer.encode(text)
        tokens = encoding.ids
        max_valid_index = len(vocab.stoi) - 1
        
        tokens = [min(token, max_valid_index) for token in tokens]
        tokens.insert(0,vocab.stoi["<cls>"])
        # Обрезаем или дополняем до длины 256
        if len(tokens) < max_len:
            padding_length = max_len - len(tokens)
            tokens = tokens + [vocab.stoi["<pad>"]] * padding_length
            return tokens
        elif len(tokens) > max_len:
            # Берем первые 128 токенов
            tokens_first_part = tokens[:128]
            # Берем последние 128 токенов
            tokens_second_part = tokens[-128:]
            # Объединяем
            tokens_first_part.extend(tokens_second_part)
            return tokens_first_part
        else:
            return tokens
    
    data["input_ids"] = data["review"].apply(pad_and_encode)
    print(data["input_ids"].head())
    # ПРОВЕРКА ДАННЫХ
    print("=== DATA VALIDATION ===")
    print(f"Vocab size: {len(vocab)}")
    
    all_input_ids = [idx for sublist in data["input_ids"] for idx in sublist]
    max_index = max(all_input_ids) if all_input_ids else 0
    min_index = min(all_input_ids) if all_input_ids else 0
    
    print(f"Max index in data: {max_index}")
    print(f"Min index in data: {min_index}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Cls token :", vocab.stoi["<cls>"])
    print(data["input_ids"].head())
    if max_index >= len(vocab):
        invalid_count = sum(1 for idx in all_input_ids if idx >= len(vocab))
        print(f"WARNING: {invalid_count} indices exceed vocab size!")
    
    return data, vocab, tokenizer
def prepareDataForImdb():
    
    tokenizer = get_tokenizer("basic_english")
    def labelFix(dt):
        if dt == 'positive':
            return 1
        else:
            return 0
    def remove_emojis(text):
        """Удаляет эмодзи и специальные символы Unicode"""
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # эмоции
                               u"\U0001F300-\U0001F5FF"  # символы & пиктограммы
                               u"\U0001F680-\U0001F6FF"  # транспорт & карты
                               u"\U0001F1E0-\U0001F1FF"  # флаги
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_urls(text):
        """Удаляет URL адреса"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_mentions_hashtags(text):
        """Удаляет упоминания и хэштеги"""
        cleaned = re.sub(r'[@#]\w+', '', text)
        return cleaned

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
    def comprehensive_text_cleaner(text, 
                             remove_urls_flag=True,
                             remove_mentions_flag=True, 
                             remove_emojis_flag=True,
                             remove_special_chars_flag=True,
                             normalize_spaces_flag=True):
        """
        Комплексная очистка текста с настройками
        """
        if not isinstance(text, str):
            return ""

        cleaned_text = text

        if remove_urls_flag:
            cleaned_text = remove_urls(cleaned_text)

        if remove_mentions_flag:
            cleaned_text = remove_mentions_hashtags(cleaned_text)

        if remove_emojis_flag:
            cleaned_text = remove_emojis(cleaned_text)

        if remove_special_chars_flag:
            # Сохраняем только буквы, цифры, пробелы и основную пунктуацию
            cleaned_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s\.\,\!\?\-\:\(\)\"]', '', cleaned_text)

        if normalize_spaces_flag:
            # Заменяем множественные пробелы на один
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
        return cleaned_text
    def clean_dataset_texts(df, text_column='review'):
        """
        Очищает тексты в датафрейме
        """

        # Базовая очистка
        df['review'] = df[text_column].apply(
            lambda x: comprehensive_text_cleaner(str(x)) if pd.notna(x) else ""
        )       

        return df
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
    data = clean_dataset_texts(data)
    def removeHtmlTags(string):
        s2 = re.sub(r"<.*?>", "", string)
        return s2
    
    data['review'] = data['review'].apply(removeHtmlTags)
    
    def gen_tokenizer(data, tokenizer):
        for text in data:
            yield tokenizer(text)
    
    gen = gen_tokenizer(data['review'], tokenizer=tokenizer)
    
    max_tokens = 28000
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
dataset = "IMDB" #"IMDB"  #"Amazon"

data, vocab, tokenizer = (prepareDataForImdbWordPeace() if dataset == "IMDB" else prepareAmazonDataset())
data['has_html'] = data['review'].str.contains(r'<[^>]+>', regex=True)

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

print("Using CPU for debugging")
#learning_rate = 9e-4 most optimal for IMDB
learning_rate = 6e-4
optim = torch.optim.AdamW(model.parameters(), lr = learning_rate)
num_epochs = 15
losses = []
val_losses = []
corrects = []
valid_result = []
val_corrects = []
val_loss, val_correct, val_total = 0.0, 0, 0
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=4, num_training_steps=15)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=6, factor=0.5)
early_stopping = es(patience=6, min_delta=0.0001)
val_acc_array = []


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
                val_losses.append(val_loss.item())
            val_corrects.append(val_correct)
            val_acc = val_correct / val_total
            val_acc_array.append(val_acc)
            print(f"Validation [{epoch}], val_loss : {val_loss}, val_correct : {val_correct}, Total {val_total}, Accuracy : {val_acc}")
            if early_stopping(val_loss, model.state_dict()):
                print("Loading best model weights")
                model.load_state_dict(early_stopping.get_best_weights())
                break
        print(f"Current LR: {optim.param_groups[0]['lr']}")
        scheduler.step()
        


if(runTrain == False):
    if dataset == "IMDB":
        model.load_state_dict(torch.load("multihead_transformer_IMDB_0.8607.pth", map_location="cpu"))
        model.to(device)
    else:
        checkpoint = torch.load("multihead_transformer_Amazon_0.87.pth", map_location=torch.device('cpu'))
    model.eval()


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
                    #"score_1sthead" : scores[0][i].cpu().numpy().tolist(),
                    #"score_2ndhead" : scores[1][i].cpu().numpy().tolist()
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
print(len(obj_for_debug))



import seaborn as sns
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
import numpy as np

def highlight_attention_html(model, text, vocab, tokenizer, device, class_label, head_idx=6):
    """
    head_idx: индекс головы (0-7). Ты сказал, что 7-я (индекс 6) самая интересная.
    """
    #model.eval()
    def pad_and_encode(text):
        max_len = 256
        encoding = tokenizer.encode(text)
        #print(type(encoding))
        tokens = encoding.ids
        #print("Cls" , [vocab.stoi["<cls>"]])
        #print(tokens)
        # Проверяем индексы
        max_valid_index = len(vocab.stoi) - 1
        
        tokens = [min(token, max_valid_index) for token in tokens]
        tokens.insert(0,vocab.stoi["<cls>"])
        # Обрезаем или дополняем до длины 256
        if len(tokens) < max_len:
            padding_length = max_len - len(tokens)
            tokens = tokens + [vocab.stoi["<pad>"]] * padding_length
            return tokens
        elif len(tokens) > max_len:
            # Берем первые 128 токенов
            tokens_first_part = tokens[:128]
            # Берем последние 128 токенов
            tokens_second_part = tokens[-128:]
            # Объединяем
            tokens_first_part.extend(tokens_second_part)
            return tokens_first_part
        else:
            return tokens
    # 1. Подготовка (как раньше)
    tokens_ids = pad_and_encode(text)
    # Обрезаем паддинги сразу, чтобы не рисовать их
    if vocab["<pad>"] in tokens_ids:
        real_len = tokens_ids.index(vocab["<pad>"])
        tokens_ids = tokens_ids[:real_len]
    else:
        real_len = len(tokens_ids)
        
    input_tensor = torch.tensor([tokens_ids], dtype=torch.long).to(device)
    
    # 2. Прогон (Хук или self-storage должен быть уже активирован)
    with torch.no_grad():
        _ = model(input_tensor)
        
    # Достаем скоры: [Batch, Heads, Seq, Seq]
    # model.attBlock2.attention.last_attention_scores
    scores = model.attBlock2.attention.last_scores[0] # [8, Seq, Seq]
    
    # 3. Фокусируемся на CLS токене (строка 0)
    # Нас интересует, как CLS (индекс 0) смотрит на все остальные слова (ось 1)
    cls_attention = scores[head_idx, 0, :real_len].numpy()
    
    # Нормализуем веса для яркости цвета (чтобы максимум был ярко-красным)
    # Можно использовать softmax или просто min-max
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-9)
    
    # Декодируем слова
    itos = vocab.get_itos()
    tokens = [itos[tid] for tid in tokens_ids]
    
    # 4. Генерируем HTML
    html_content = f"<h3>Attention Map (Head {head_idx+1})</h3>"
    html_content += f"<p><b>Review Score:</b> {torch.sigmoid(model(input_tensor)[0]).item():.4f}</p>"
    html_content += '<div style="border:1px solid #ccc; padding: 10px; line-height: 2.0; font-family: sans-serif;">'
    
    for word, weight in zip(tokens, cls_attention):
        # Цвет: Красный с прозрачностью (alpha) равной весу внимания
        # Чем важнее слово, тем насыщеннее фон
        alpha = weight * 0.8 + 0.1 # чуть сдвигаем, чтобы совсем бледные тоже было видно
        html_content += f'<span style="background-color: rgba(255, 0, 0, {alpha:.2f}); padding: 2px; margin: 1px; border-radius: 3px;">{word}</span> '
        
    html_content += '</div>'
    html_content += f"</br>"

    return html_content


def highlight_attention_html_heads(model, text_sample, vocab, tokenizer, device, class_, count_heads):
    filename = f"attention_{class_}.html"
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(count_heads):
            html_content = highlight_attention_html(model, text_sample, vocab, tokenizer, device, class_, head_idx=i)
            f.write(html_content)

# === ЗАПУСК ===
text_sample = "this movie was terrible acting was bad but the music was nice" 
text_sample_pos = data[data["label"] == 1]["review"].iloc[5]



highlight_attention_html_heads(model, text_sample, vocab, tokenizer, device, "negative", count_heads=8)
highlight_attention_html_heads(model, text_sample_pos, vocab, tokenizer, device, "positive", count_heads=8)



# Смотрим 7-ю голову (индекс 6), которая тебе понравилась
#highlight_attention_html(model, text_sample, vocab, tokenizer, device, "negative", head_idx=6)#

## Смотрим 3-ю голову (индекс 2), которая "странная"
#highlight_attention_html(model, text_sample, vocab, tokenizer, device, "negative", head_idx=2)
#highlight_attention_html(model, text_sample_pos, vocab, tokenizer, device, "positive", head_idx=6)
#highlight_attention_html(model, text_sample_pos, vocab, tokenizer, device, "positive", head_idx=2)

all_probs_np = np.concatenate(all_probs)
all_labels_np = np.concatenate(all_labels)
all_probs_np_squeezed = np.concatenate(probs)
all_labels_np_squeezed = np.concatenate(labels_probs)

if(runTrain):
    torch.save(model.state_dict(), f"multihead_transformer_{dataset}_{accuracy:.4f}.pth")

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
print(f"False negatives: {len(false_negatives)}")
print(f"False positives: {len(false_positives)}")

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

import model_train_tools.DebuggerTools as db
import tools as vc

debug = db.TransformerDebugger(vocab=vocab, model=model)
debug.build_word_emb_metrics(tokens_spec = ["<unk>", '.', '<pad>', '\'', ',', '!'],
    token_words =['of', 'movie', 'good', 'great', 'like', 'story'])

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
for item in counter_Ngrams.most_common(10):
    arr_items_neg.append({"ngram_text": f'{vocab_itos[item[0][0]]} {vocab_itos[item[0][1]]} {vocab_itos[item[0][2]]}', "count": item[1], "color" : "red"})
for item in counter_PosNrgams.most_common(10):
    arr_items_pos.append({"ngram_text": f'{vocab_itos[item[0][0]]} {vocab_itos[item[0][1]]} {vocab_itos[item[0][2]]}', "count": item[1], "color" : "green"})#


arr_temp = []
arr_temp.extend(arr_items_neg)
arr_temp.extend(arr_items_pos)#




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
#uncertrain = [i for i in obj_for_debug if i["uncertrain"] == True]
#
#print("Uncertrain cases:")
#for i in uncertrain:
#    print(i)


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


import csv

with open("result_csv.csv", 'w', newline='', encoding='utf-8') as csvfile:
    csv_file_writer = csv.writer(csvfile)
    for i in range(len(obj_for_debug)):
        item = obj_for_debug[i]
        result = [item["text"],item["color_metric"],item["textLen"]]
        csv_file_writer.writerow(result) 



plt.show()