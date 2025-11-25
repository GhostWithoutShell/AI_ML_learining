import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def main(embedings):
    X_evc = TSNE(n_components=2, random_state=0, metric="euclidean", perplexity=50, max_iter=5000).fit_transform(embedings)
    X_cos = TSNE(n_components=2, random_state=0, metric="cosine", perplexity=50).fit_transform(embedings)
    #print(X.shape)
    return X_evc, X_cos
def build_plot_data(X):
    plot_data = []
    
def extract_nrgams(token_list, n):
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n > len(token_list):
        raise ValueError("n cannot be greater than the length of the token list")

    ngram = []
    for i in range(len(token_list) - n + 1):
        temp_list = (token_list[i:i+n])
        
        ngram.append(tuple(temp_list))
    return ngram