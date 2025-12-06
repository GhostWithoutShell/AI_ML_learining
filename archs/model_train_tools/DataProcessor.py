
from tokenize import Whitespace
import pandas as pd
import re

from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

class DataBulder:

    def getDataFromCsv():
        pass
    def getDataFromExcel():
        pass

class DataBuilderImdb(DataBulder):
    def getDataFromCsv(self, path):
        data = pd.read_csv(path)
        return data
        
    def cleanTextFromTrash(self, data, parameters):
        column = parameters["textsColumn"]
        #data.columns = [parameters["labelsColumn"], parameters["textsColumn"]]
        data[column] = data[column].apply(self._removeHtmlTags)
        data[column] = data[column].apply(self._comprehensive_text_cleaner)    
        return data
    def applyLabelFix(self, data, parameters):
        column = parameters["labelsColumn"]
        print("Col" ,column)
        data[column] = data[column].apply(self._labelFix)
        return data

    def _labelFix(self, dt):
        if dt == 'positive':
            return 1
        else:
            return 0
        
    def _remove_emojis(self, text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # эмоции
                               u"\U0001F300-\U0001F5FF"  # символы & пиктограммы
                               u"\U0001F680-\U0001F6FF"  # транспорт & карты
                               u"\U0001F1E0-\U0001F1FF"  # флаги
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def _remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def _remove_mentions_hashtags(self, text):
        cleaned = re.sub(r'[@#]\w+', '', text)
        return cleaned

    def _removeHtmlTags(self, string):
        s2 = re.sub(r"<.*?>", "", string)
        return s2 
    def _comprehensive_text_cleaner(self, text):
        if not isinstance(text, str):
            return ""
            
        cleaned_text = text
        cleaned_text = self._remove_urls(cleaned_text)
        cleaned_text = self._remove_mentions_hashtags(cleaned_text)
        cleaned_text = self._remove_emojis(cleaned_text)
        cleaned_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s\.\,\!\?\-\:\(\)\"]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

        
    
class TokenizatorProcessing:
    def prepareTokenizator(self, tokenizator_name):
        pass
    def getData(self, tokenizator_name):
        pass

class TokenizatorProcessingWordPeace(TokenizatorProcessing):
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
        
    def __init__(self, max_length, special_tokens, vocab_size=30000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.__prepareTokenizator()
    def __prepareTokenizator(self, tokenizator_name):
        self.tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        
    def __get_training_corpus(self, len_text, dt, column_with_text):
        for i in range(0, len_text, 1000):
            yield dt[column_with_text][i:i+1000].tolist()
    
    def getData(self, tokenizator_name):
        pass
    def padAndEncode(self, text, use_slice = False):
        encoded = self.tokenizer.encode(text)
        token_ids = encoded.ids
        if use_slice:
            token_ids = token_ids[:self.max_length]
        else:
            first_part = token_ids[:self.max_length//2]
            second_part = token_ids[self.max_length//2:]
            token_ids = first_part + second_part
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids += [0] * padding_length
        return token_ids
    def prepareVocab(self, dt, column_with_text):
        len_text = len(dt)
        self.tokenizer.train_from_iterator(self.__get_training_corpus(len_text, dt, column_with_text), trainer=self.trainer)
        return self.__getVocab()
    def __getVocab(self):
        return self.CustomVocab(self.tokenizer)

