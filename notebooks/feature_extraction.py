import os
import re
import gzip
from nltk import  word_tokenize
from pymystem3 import Mystem
import numpy as np
from sklearn import feature_extraction
from utils import Word2vecProcessor
from scipy.spatial.distance import cosine
from tqdm import tqdm

with gzip.open("../data/external/news_upos_cbow_300_2_2017.bin.gz") as mf:
    word2vec = Word2vecProcessor(mf)

_m = Mystem()
with open("../data/external/stopwords.txt") as stop_file:
    stopwords = stop_file.readlines();
    stopwords = [s.replace("\n", "") for s in stopwords]

stopwords.append("http")
stopwords.append("https")

def normalize_word(w):
    return _m.lemmatize(w)[0].lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in word_tokenize(s) if t.isalpha()]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    s_clean = " ".join(get_tokenized_lemmas(s))
    return s_clean

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in stopwords]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)


def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


# def refuting_features(headlines, bodies):
#     _refuting_words = [
#         'fake',
#         'fraud',
#         'hoax',
#         'false',
#         'deny', 'denies',
#         'not',
#         'despite',
#         'nope',
#         'doubt', 'doubts',
#         'bogus',
#         'debunk',
#         'pranks',
#         'retract'
#     ]
#     X = []
#     for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
#         clean_headline = clean(headline)
#         clean_headline = get_tokenized_lemmas(clean_headline)
#         features = [1 if word in clean_headline else 0 for word in _refuting_words]
#         X.append(features)
#     return X


# def polarity_features(headlines, bodies):
#     _refuting_words = [
#         'fake',
#         'fraud',
#         'hoax',
#         'false',
#         'deny', 'denies',
#         'not',
#         'despite',
#         'nope',
#         'doubt', 'doubts',
#         'bogus',
#         'debunk',
#         'pranks',
#         'retract'
#     ]
#
#     def calculate_polarity(text):
#         tokens = get_tokenized_lemmas(text)
#         return sum([t in _refuting_words for t in tokens]) % 2
#     X = []
#     for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
#         clean_headline = clean(headline)
#         clean_body = clean(body)
#         features = []
#         features.append(calculate_polarity(clean_headline))
#         features.append(calculate_polarity(clean_body))
#         X.append(features)
#     return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body.split(" "):
            grams_hits += 1
        if gram in text_body.split(" ")[:50]:
            grams_early_hits += 1
        if gram in text_body.split(" ")[:25]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body.split(" ")[:50]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features

def cosine_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        cos = cosine(word2vec.text_vector(headline), word2vec.text_vector(body))
        X.append(cos)
    return X

def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in headline.split(" "):
            if headline_token in body.split(" "):
                bin_count += 1
            if headline_token in body.split(" ")[:50]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(headline.split(" ")):
            if headline_token in body.split(" "):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph
        features = []
        features = append_chargrams(features, headline, body, 2)
        features = append_chargrams(features, headline, body, 8)
        features = append_chargrams(features, headline, body, 4)
        features = append_chargrams(features, headline, body, 16)
        features = append_ngrams(features, headline, body, 2)
        features = append_ngrams(features, headline, body, 3)
        features = append_ngrams(features, headline, body, 4)
        features = append_ngrams(features, headline, body, 5)
        features = append_ngrams(features, headline, body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        X.append(binary_co_occurence(clean_headline, clean_body)
                 + binary_co_occurence_stops(clean_headline, clean_body)
                 + count_grams(clean_headline, clean_body))

    return X
