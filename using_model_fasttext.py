from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np

model = KeyedVectors.load('./model/fasttext_gensim.model')

for word in model.most_similar(u"công_nghệ"):
    print(word[0])