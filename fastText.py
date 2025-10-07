import os
import pandas as pd
import string
from pyvi import ViTokenizer
from gensim.models.fasttext import FastText

# path data
pathdata = './datatrain.txt'

def read_data(path):
    traindata = []
    sents = open(pathdata, 'r').readlines()
    for sent in sents:
        traindata.append(sent.split())
    return traindata

if __name__ == '__main__':
    train_data = read_data(pathdata)

    model_fasttext = FastText(vector_size=150, window=10, min_count=2, workers=4, sg=1, epochs=5)
    model_fasttext.build_vocab(train_data)
    model_fasttext.train(train_data, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.epochs)

    model_fasttext.wv.save("./model/fasttext_gensim.model")