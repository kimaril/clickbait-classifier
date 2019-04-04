import numpy as np
from tqdm import tqdm_notebook
from gensim.models import KeyedVectors
from nltk import word_tokenize
from scipy.spatial.distance import cosine
from pymystem3 import Mystem

class Word2vecProcessor(object): # sberbank ai exmple code
    """Объект для работы с моделью word2vec сходства слов"""

    def __init__(self, w2v_model_file):
        self.mystem = Mystem()
        self.word2vec = KeyedVectors.load_word2vec_format(w2v_model_file, binary=True)
        self.lemma2word = {word.split('_')[0]: word for word in self.word2vec.index2word}

    def word_vector(self, word):
        lemma = self.mystem.lemmatize(word)[0]
        word = self.lemma2word.get(lemma)
        return self.word2vec[word] if word in self.word2vec else None

    def text_vector(self, text):
        """Вектор текста, получается путем усреднения векторов всех слов в тексте"""
        word_vectors = [
            self.word_vector(token)
            for token in word_tokenize(text.lower())
            if token.isalpha()
            ]
        word_vectors = [vec for vec in word_vectors if vec is not None]
        return np.mean(word_vectors, axis=0)

    # def distance(text1, text2):
    #     if text1 is None or text2 is None:
    #         return 2
    #     vec1 = self.text_vector(text1)
    #     vec2 = self.text_vector(text2)
    #     return cosine(vec1, vec2)

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords
        self.stopwords.append("https")
        self.stopwords.append("http")

    def beautify(self, text):
        sent = [self.mystem.lemmatize(word)[0] for word in
                word_tokenize(text.lower()) if word.isalpha()]
        return " ".join([s for s in sent if s not in self.stopwords])
