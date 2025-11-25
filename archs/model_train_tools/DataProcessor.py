
import pandas as pd
import re
class DataBulder:

    def getDataFromCsv():
        pass
    def getDataFromExcel():
        pass

class DataBuilderImdb(DataBulder):
    def getDataFromCsv(self, path):
        data = pd.read_csv('D:/MyFiles/Datasets/IMDB/IMDB Dataset.csv')
        data.columns = ['review', 'label']
        data['label'] = data['label'].apply(self.labelFix)
        data['review'] = data['review'].apply(self.removeHtmlTags)
        data['review'] = data['review'].apply(self.comprehensive_text_cleaner)    

    def labelFix(self, dt):
        if dt == 'positive':
            return 1
        else:
            return 0
    def remove_emojis(self, text):
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
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_mentions_hashtags(text):
        cleaned = re.sub(r'[@#]\w+', '', text)
        return cleaned

    def removeHtmlTags(string):
        s2 = re.sub(r"<.*?>", "", string)
        return s2 
    def comprehensive_text_cleaner(self, text):
        if not isinstance(text, str):
            return ""
            
        cleaned_text = text
        cleaned_text = self.remove_urls(cleaned_text)
        cleaned_text = self.remove_mentions_hashtags(cleaned_text)
        cleaned_text = self.remove_emojis(cleaned_text)
        cleaned_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s\.\,\!\?\-\:\(\)\"]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    
    
class TokenizatorProcessing:
    def prepareTokenizator(self, tokenizator_name):
        pass
    def getData(self, tokenizator_name):
        pass

class TokenizatorProcessingWordPeace(TokenizatorProcessing):
    def prepareTokenizator(self, tokenizator_name):
        pass
    def getData(self, tokenizator_name):
        pass
