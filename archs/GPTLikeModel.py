import torch.nn as nn
import seaborn as sns
import numpy as np
import re
import torch
import math
import model_train_tools.DataProcessor as dp
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True

class AttentionLayer(nn.Module):
    def __init__(self, size_kernel, num_heads, pad_index):
        super().__init__()
        self.size_kernel = size_kernel
        self.num_heads = num_heads
        self.pad_index = pad_index
        self.head_dim = size_kernel // num_heads
        self.last_attention_scores = None
        self.key = nn.Linear(size_kernel, size_kernel)
        self.value = nn.Linear(size_kernel, size_kernel)
        self.query = nn.Linear(size_kernel, size_kernel)
        self.projection = nn.Linear(size_kernel, size_kernel)
    def forward(self, x, input_ids):
        batch_size, seq_len, _ = x.size()
        x_k = self.key(x)
        x_q = self.query(x)
        x_v = self.value(x)
        x_k = x_k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x_v = x_v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x_q = x_q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x_v_tr = x_v.transpose(1,2)
        x_q_tr = x_q.transpose(1,2)
        x_k_tr = x_k.transpose(1,2)
        #print(x_q_tr.shape)
        #print(x_k_tr.transpose(-1,-2).shape)
        scores = torch.matmul(x_q_tr, x_k_tr.transpose(-1,-2))
        # mask
        padding_mask = (input_ids != self.pad_index).unsqueeze(1).unsqueeze(2)
        tri_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0).bool()
        final_mask = padding_mask & tri_mask

        scores = scores/math.sqrt(self.head_dim)
        min_value = torch.finfo(scores.dtype).min
        ## for resolve problem with autoscale 
        attention_scores = scores.masked_fill(final_mask == 0, min_value)

        attention_scores = torch.softmax(attention_scores, dim=-1)

        self.last_attention_scores = attention_scores.detach().cpu()

        result = torch.matmul(attention_scores, x_v_tr)
        result = result.transpose(1,2).contiguous()
        result = result.view(batch_size, seq_len, self.size_kernel)
        x = self.projection(result)
        return x

class FeedForward(nn.Module):
    def __init__(self, size_kernel, expansion=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size_kernel, size_kernel*expansion),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(size_kernel*expansion, size_kernel)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, size_kernel, num_heads, pad_index):
        super().__init__()
        self.attention = AttentionLayer(size_kernel, num_heads, pad_index)
        
        
        self.norm = nn.LayerNorm(size_kernel)
        self.norm_ff = nn.LayerNorm(size_kernel)
        self.dropout = nn.Dropout(0.2)
        self.ff = FeedForward(size_kernel=size_kernel)

    def forward(self, x, input_ids):
        x_norm = self.norm(x)
        x_att = self.attention(x_norm, input_ids)
        x = self.dropout(x_att) + x
        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x
    

    
class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, size_kernel, num_heads, num_layers, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, size_kernel, padding_idx=pad_index)
        self.pos_emb = nn.Embedding(512, size_kernel)
        self.norm = nn.LayerNorm(size_kernel)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(size_kernel, num_heads, pad_index) for _ in range(num_layers)
        ])        
        self.drop = nn.Dropout(0.2)
        self.fc_out = nn.Linear(size_kernel, vocab_size, bias=False)
        self.embedding.weight =self.embedding.weight
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(input_ids.device)
        x = self.embedding(input_ids) + self.pos_emb(positions)
        x = self.norm(x)
        for block in self.attention_blocks:
            x = block(x, input_ids)
        logits = self.fc_out(self.drop(x))
        return logits
    

class GptLikeDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long)



def train_step(optim, criterion, model, input_ids, vocab_size):
    train_input = input_ids[:, :-1]
    train_targets = input_ids[:, 1:]
    outputs = model(train_input)
    outputs = outputs.reshape(-1, vocab_size)
    train_targets = train_targets.reshape(-1)
    loss = criterion(outputs, train_targets)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss.item()

dt = dp.DataBuilderImdb()
data = dt.getDataFromCsv("D:\MyFiles\Datasets\Russian_jokes\jokes.csv\jokes.csv")
params = {
    "textsColumn": "text",
}
#data = dt.getDataFromCsv("D:\MyFiles\Datasets\IMDB\IMDB Dataset.csv")
#params = {
#    "textsColumn": "review",
#    "labelsColumn": "sentiment"
#}
data = dt.cleanTextFromTrash(data, params)
#data = dt.applyLabelFix(data, params)
print(data.head())

num_epochs = 6
batch_size = 16
learning_rate = 0.001
vocab_size = 15000


print("Len dataset" ,len(data))

#data = data[data["rating"] > 1]
data_ = data[['text']]

tokenizerWrap = dp.TokenizatorProcessingWordPeace(max_length=256, special_tokens=["<unk>", "<pad>"], vocab_file_name="russian_joke_vocab_15.json", vocab_size=vocab_size)
vocab = tokenizerWrap.prepareVocab(data_, column_with_text="text")
data_["input_ids"] = data_["text"].apply(lambda x: tokenizerWrap.padAndEncode(x, vocab=vocab, use_first_and_second_part=False))
data_.columns = ['review', 'input_ids']

print(data.head())
print(len(data))
import torch.nn.functional as F


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def visualize_attention(model, text, vocab, tokenizerWrap, device, layer_num=0, head_num=0):
    model.eval()
    
    # 1. Подготовка данных
    # Используем твой токенизатор
    input_ids = tokenizerWrap.padAndEncode(text, vocab=vocab, use_first_and_second_part=False)
    
    # Обрезаем паддинги, чтобы картинка была чистой (без кучи <pad>)
    pad_id = vocab.get_stoi()["<pad>"]
    if pad_id in input_ids:
        real_len = input_ids.index(pad_id)
        input_ids = input_ids[:real_len]
    else:
        real_len = len(input_ids)

    # Декодируем токены обратно в слова для подписей осей
    tokens = tokenizerWrap.tokenizer.decode(input_ids).split() # Или свой метод получения списка слов
    # Если decode возвращает строку, можно попробовать vocab.lookup_tokens(input_ids)
    
    # Создаем тензор
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 2. Прогон через модель
    with torch.no_grad():
        _ = model(input_tensor)

    # 3. Достаем сохраненные скоры
    # model.attention_blocks[layer_num] -> это i-й блок
    # .attention -> это твой AttentionBlock
    # .attention -> это твой AttentionLayer (проверь вложенность в своем коде!)
    # Если у тебя AttentionBlock.attention = AttentionLayer, то путь такой:
    
    # Получаем матрицу внимания [Batch, Heads, Seq_Len, Seq_Len]
    # Берем [0] (первый элемент батча) -> [Heads, Seq, Seq]
    scores = model.attention_blocks[layer_num].attention.last_attention_scores[0]
    
    # Берем конкретную голову
    head_scores = scores[head_num] # [Seq, Seq]
    
    # Обрезаем матрицу под реальную длину (без паддингов)
    head_scores = head_scores[:real_len, :real_len]

    # 4. Рисуем
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        head_scores, 
        xticklabels=tokens, 
        yticklabels=tokens, 
        cmap="viridis", # или "Blues", "Reds"
        annot=False,    # Если True, будет писать цифры (грязно для больших матриц)
        square=True
    )
    plt.title(f"Layer {layer_num+1}, Head {head_num+1}")
    plt.xlabel("Key (на кого смотрим)")
    plt.ylabel("Query (кто смотрит)")
    plt.show()





def generate_sample(model, prompt, max_tokens=30, temperature=0.8, top_k=50):
    model.eval()
    ids = tokenizerWrap.padAndEncode(prompt, vocab=vocab, use_first_and_second_part=False)
    pad_id = vocab.get_stoi()["<pad>"]
    if pad_id in ids:
        ids = ids[:ids.index(pad_id)]
        
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            
            next_token_logits = logits[:, -1, :]
            
            next_token_logits = next_token_logits / temperature
            
            if top_k > 0:
                
                top_values, _ = torch.topk(next_token_logits, top_k)
                min_val = top_values[:, -1]    
                next_token_logits[next_token_logits < min_val] = -float('Inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Это "бросок кубика"
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
    result_ids = input_ids[0].tolist()
    return tokenizerWrap.tokenizer.decode(result_ids)


import sys

model_state_dict_file_name = 'my_first_gpt_russian.pth'
pad_index = vocab.get_stoi()["<pad>"]
real_vocab_size = len(vocab.get_stoi())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(type):
    if type == "train":
        train, test = train_test_split(data_, test_size=0.2, random_state=42)
        
        train_dataset = GptLikeDataset(train["input_ids"].tolist())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = GptLikeDataset(test["input_ids"].tolist())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        import time
        print(data_.head())  
        print(len(data_["input_ids"].iloc[0]))
        scaler = torch.cuda.amp.GradScaler()

        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_index)
        losses = 0
        loss_values = []
        model = GPTLikeModel(vocab_size=real_vocab_size, size_kernel=256, num_heads=8, num_layers=3, pad_index=pad_index).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr= learning_rate)
        start = time.time()
        print("#### START TRAIN LOOP")
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            model.train()
            losses = 0
            accumulation_steps = 4  # Хотим эмулировать батч 16 * 4 = 64
            model.zero_grad()
            for i, input_ids in enumerate(train_loader):
                input_ids = input_ids.to(device)
                # берем все кроме последнего токена
                train_trargets = input_ids[:, 1:]
                # берем все кроме первого токена
                train_input = input_ids[:, :-1]
                # получаестя [a,b,c], [b,c,d] это нужно для того чтобы мы могли парралельно сравнивать предсказания модели с правильными ответами
                # иначе бы пришлось делать цикл по всем токенам, по итогу мы не берем последний токен т.к для него нет пары, а сравниваем 1ый токен со вторым и т.д
                #optim.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(train_input)
                    outputs = outputs.reshape(-1, len(vocab.get_itos()))
                    train_trargets = train_trargets.reshape(-1)
                    loss = criterion(outputs, train_trargets)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                losses += loss.item()
            avg_loss = losses / len(train_loader)

            model.eval()
            total_val_losses = 0
            correct_tokens = 0
            total_tokens = 0
            with torch.no_grad():
                for input_ids_val in test_loader:
                    input_ids_val = input_ids_val.to(device)
                    test_targets = input_ids_val[:, 1:]
                    test_input = input_ids_val[:, :-1]

                    outputs = model(test_input)
                    loss = criterion(outputs.reshape(-1, len(vocab.get_itos())), test_targets.reshape(-1))

                    total_val_losses += loss.item()
                    prediction = torch.argmax(outputs, dim=-1)
                    mask = (test_targets != pad_index)

                    correct = (prediction == test_targets) & mask
                    correct_tokens += correct.sum().item()
                    total_tokens += mask.sum().item()

                avg_val_loss = total_val_losses / len(test_loader)
                val_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0000

            print(f'Epoch {epoch+1} | Val avg loss {avg_val_loss:.4f} | Val acc {val_accuracy:.4f}')
        end = time.time() - start
        torch.save(model.state_dict(), model_state_dict_file_name)
        print("Model saved successfully!")
        print("Train loop time :", end)
    elif type == "debug":
        print("\n=== AI CHAT (type 'exit' to stop) ===")
        model = GPTLikeModel(vocab_size=real_vocab_size, size_kernel=256, num_heads=8, num_layers=3, pad_index=pad_index).to(device)
        model.load_state_dict(torch.load(model_state_dict_file_name))
        model = model.to(device)

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Генерируем
            response = generate_sample(
                model, 
                user_input, 
                max_tokens=30, 
                temperature=0.8, 
                top_k=50
            )
            response = response.replace(user_input, "") 

            print(f"AI: {response}") 
            visualize_attention(model, response, vocab, tokenizerWrap, device, layer_num=0, head_num=0)
            visualize_attention(model, response, vocab, tokenizerWrap, device, layer_num=2, head_num=7)

args = sys.argv[1:]
type = args[0]

main(type=type)