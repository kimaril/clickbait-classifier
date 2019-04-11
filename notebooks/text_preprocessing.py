import numpy as np
import re
from nltk import word_tokenize
from scipy.spatial.distance import cosine
from pymystem3 import Mystem
import wget
import zipfile
import json
from gensim.models import KeyedVectors

class Preprocessor:

    def __init__(self, stopwords):
        self.mystem = Mystem()
        self.stopwords = stopwords

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords

    def remove_url(self, text):
        return re.sub(r'[\s]*https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

    def remove_non_alphabetic(self, text):
        return " ".join([w for w in word_tokenize(text.lower()) if w.isalpha()])

    def normalize(self, word):
        return self.mystem.lemmatize(word)[0]

    def normalize_text(self, text):
        if isinstance(text, str):
            return [self.normalize(w) for w in text.split()]
        elif isinstance(text, list):
            return [self.normalize(w) for w in text]
        else:
            raise TypeError("\'text\' argument must be of \'list\' or \'str\' type!")

    def remove_stopwords(self, text, normalize_text):
        if normalize_text==False:
            return " ".join([w for w in text.split() if self.normalize(w) not in self.stopwords])
        else:
            return " ".join([w for w in self.normalize_text(text) if w not in self.stopwords])

    def beautify(self, text, normalize_text=False):
        try:
            cleaned = self.remove_non_alphabetic(self.remove_url(text))
            cleaned_sw = self.remove_stopwords(cleaned, normalize_text=normalize_text)
            return cleaned_sw
        except TypeError:
            raise

class Word2VecProcessor:

    def __init__(self, model_path=None, model_url=None, download=False):
        # mystem = Mystem()
        if download:
            model_file = wget.download(model_url, out="../../data/external")
        else:
            model_file = model_path
        with zipfile.ZipFile(model_file, 'r') as archive:
            self.meta = json.load(archive.open("meta.json"))
            stream = archive.open('model.bin')
            self.word2vec = KeyedVectors.load_word2vec_format(stream, binary=True)

        if self.meta["corpus"][0]["tagset"] is None:
            self.lemma2word = self.word2vec.index2word
        else:
            self.lemma2word = {word.split('_')[0]: word for word in self.word2vec.index2word}

    def word_vector(self, word):
        word = self.lemma2word.get(word)
        return self.word2vec[word] if word in self.word2vec else None

    def text_matrix(self, text):
        word_vectors = [
            self.word_vector(token)
            for token in word_tokenize(text.lower())
            if token.isalpha()
            ]
        word_vectors = [vec for vec in word_vectors if vec is not None]
        return word_vectors

    def text_vector(self, text):
        word_vectors = self.text_matrix(text)
        return np.mean(word_vectors, axis=0)

    def distance(self, text1, text2):
        return cosine(self.text_vector(text1), self.text_vector(text2))

class FastTextProcessor:
    def __init__(self, model_path, meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        self.word2vec = KeyedVectors.load(model_path)

    def word_vector(self, word):
        return self.word2vec[word] if word in self.word2vec.index2word else None

    def text_matrix(self, text):
        word_vectors = [
            self.word_vector(token)
            for token in word_tokenize(text.lower())
            if token.isalpha()
            ]
        # oov_count = sum(1 for _ in filter(None.__ne__, word_vectors))
        # print("Out Of Vocabulary: {}".format(oov_count))
        word_vectors = [vec for vec in word_vectors if vec is not None]
        return word_vectors

    def text_vector(self, text):
        word_vectors = self.text_matrix(text)
        return np.mean(word_vectors, axis=0)
