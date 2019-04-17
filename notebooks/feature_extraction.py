import os
import re
import gzip
from nltk import  word_tokenize
from pymystem3 import Mystem
import numpy as np
from text_preprocessing import Word2VecProcessor, Preprocessor
from scipy.spatial.distance import cosine
from tqdm import tqdm
from rouge import Rouge


word2vec = Word2VecProcessor(model_path="../data/external/184.zip")


with open("../data/external/stopwords.txt") as stop_file:
    stopwords = stop_file.readlines();
    stopwords = [s.replace("\n", "") for s in stopwords]
p = Preprocessor(stopwords=stopwords)

def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)


def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = p.beautify(headline)
        clean_body = p.beautify(body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def bait_features(headlines, bodies):
    _indicating_words = [
        'фейк',
        'шок',
        'сесть в лужу',
        'комментировать',
        'прокомментировать',
        'комментарий',
        'отреагировать',
        'подробность',
        'рассказать',
        'конец света',
        'жестко',
        'издеваться',
        'тайна'
    ]
    X = []
    for i, headline in tqdm(enumerate(headlines)):
        clean_headline = p.beautify(headline)
        features = [1 if word in clean_headline else 0 for word in _indicating_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _indicating_words = [
        'фейк',
        'шок',
        'сесть в лужу',
        'комментировать',
        'прокомментировать',
        'комментарий',
        'отреагировать',
        'подробность',
        'рассказать',
        'конец света',
        'жестко',
        'издеваться',
        'тайна'
    ]

    def calculate_polarity(text):
        return sum([t in _indicating_words for t in text]) % 2

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = p.beautify(headline)
        clean_body = p.beautify(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input
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
    grams = [" ".join(x) for x in chargrams(" ".join(text_headline), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:50]:
            grams_early_hits += 1
        if gram in text_body[:25]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [" ".join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:50]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def cosine_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        cos = word2vec.distance(" ".join(headline), " ".join(body))
        X.append(cos)
    return X

def rouge_features(headlines, bodies):
    X = []
    rouge = Rouge()
    fails = 0
    for h, b in tqdm(zip(headlines, bodies)):
        rouge_values = []
        try:
            scores = rouge.get_scores(h, b)
            rouge_values += [scores[0]['rouge-1']['f']]
            rouge_values += [scores[0]['rouge-2']['f']]
            rouge_values += [scores[0]['rouge-l']['f']]
        except:
            rouge_values = [0,0,0]
            fails += 1
        X += [rouge_values]
    print("Number of fails: {}".format(fails))
    return X

def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in headline:
            if headline_token in body:
                bin_count += 1
            if headline_token in body[:50]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in headline:
            if headline_token in body:
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
        clean_headline = p.beautify(headline)
        clean_body = p.beautify(body)
        X.append(binary_co_occurence(clean_headline, clean_body)
                 + binary_co_occurence_stops(clean_headline, clean_body)
                 + count_grams(clean_headline, clean_body))

    return X
