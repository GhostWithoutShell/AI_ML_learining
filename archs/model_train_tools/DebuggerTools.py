from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cosine
import torch
import tools as vc
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
#import .vector_checking as vc


class Debugger:
    def __init__(self, vocab, model):
        self.vocab = vocab
        self.model = model
    def build(self):
        return self#

# Debuger builder for transformer model
class TransformerDebugger(Debugger):
    def __init__(self, vocab, model):
        super().__init__(vocab, model)
    def build_word_emb_metrics(self, tokens_spec, token_words):
        print("Debug info word embeddings.....")
        #tokens_spec = ["<unk>", '.', '<pad>', '\'', ',', '!']
        #token_words =['of', 'movie', 'good', 'great', 'like', 'story']
        weights_spec = []
        weights_word = []
        with torch.no_grad():
            for i in tokens_spec:
                weights_spec.append(self.model.emb.weight[self.vocab[i]].data.cpu().detach().numpy())
            for i in token_words:
                weights_word.append(self.model.emb.weight[self.vocab[i]].data.cpu().detach().numpy())
        arr_cos_words = []
        arr_cos_spec = []#
        for i in range(len(weights_word)):
            result = cosine(weights_spec[0], weights_word[i])
            print("Cosine between <unk> and", token_words[i], ":", result)
            arr_cos_words.append(cosine(weights_spec[0], weights_word[i]))
        for i in range(1, len(weights_spec)):
            result = cosine(weights_spec[0], weights_spec[i])
            print("Cosine between <unk> and", tokens_spec[i], ":", result)
            arr_cos_spec.append(cosine(weights_spec[0], weights_spec[i]))#
        print("Special tokens weights:", cosine(weights_spec[0], weights_word[0]))
        print("Special tokens weights:", cosine(weights_spec[0], weights_word[1]))#
        print("Norm of spec tok", torch.norm(torch.tensor(weights_spec[0])))
        print("Norm of of word", torch.norm(torch.tensor(weights_word[0])))#
        return self
    def build_conf_matrix(self, all_labels, all_preds):
        self.conf_matrix = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_preds))
        return self
    def build_precision_recall(self):
        self.precision = self.conf_matrix[1][1] / (self.conf_matrix[1][1] + self.conf_matrix[0][1])
        self.recall = self.conf_matrix[1][1] / (self.conf_matrix[1][1] + self.conf_matrix[1][0])
        return self
    def build_ngrams_analysis(self, tokens_list, tokens_list_pos, n = 3):
        
        vocab_itos = self.vocab.get_itos()
        result_ngrams = vc.extract_nrgams(tokens_list, n = n)
        result_ngrams_pos = vc.extract_nrgams(tokens_list_pos, n = n)
        self.counter_Ngrams = Counter(result_ngrams)
        self.counter_PosNrgams = Counter(result_ngrams_pos)
        arr_items_neg = []
        arr_items_pos = []

        counter_Ngrams = Counter(result_ngrams)
        counter_PosNrgams = Counter(result_ngrams_pos)
        def getValueForNgramm(item, count_n = 2):
            result = ""
            result += "ngram_text"
            result += f'{vocab_itos[item[0][0]]}'
            result += f'{vocab_itos[item[0][1]]}'
            if count_n == 3:
                result += f'{vocab_itos[item[0][2]]}'
            if count_n == 4:
                result += f'{vocab_itos[item[0][3]]}'
            return result

        for item in counter_Ngrams.most_common(20):
            arr_items_neg.append({"ngram_text":f'{getValueForNgramm(item, 3)}', "count" : {item[1]}, "color" : "red"})
        for item in counter_PosNrgams.most_common(20):
            arr_items_pos.append({"ngram_text":f'{getValueForNgramm(item, 3)}', "count" : {item[1]}, "color" : "red"})
        result = {}
        result["items_pos"] = arr_items_pos
        result["items_neg"] = arr_items_neg
        
        return arr_items_pos, arr_items_neg
    def buildMetrics(self, metricsObject):
        metrics_arr = metricsObject["metrics"]
        for i in len(metrics_arr):
            if metrics_arr[i] == "word_emp":
                build_word_emb_metrics(metricsObject["token_spec"], metricsObject["token_words"])


class PlotWorker:
    def __init__(self, metricsProcessor, model_values):
        self.metricsProcessor = metricsProcessor
        self.model_values = model_values
    def build_Plot(self, plot_names):
        pass

class TransformerWorker(PlotWorker):
    def build_Plot(self, plot_names):
        for i in plot_names:
            if i == "roc":
                self.show_roc_auc_graph();
            elif i == "nrgam_conf":
                self.show_ngram_conf();
        return super().build_Plot(plot_names)
    def show_roc_auc_graph(self):
        fpr, tpr, roc_auc = self.metricsProcessor()
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
    def show_plot_with_distrib_of_percentage(self, all_probs_sigmoid, optimal_threshold=0.65):
        plt.figure(figsize=(10, 5))
        plt.hist(all_probs_sigmoid, bins=50, edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Model Predictions')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (0.5)')
        plt.axvline(x=optimal_threshold, color='g', linestyle='--', label='Optimal threshold (0.65)')
        plt.legend()
        plt.show()