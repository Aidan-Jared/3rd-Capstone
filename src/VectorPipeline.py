import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import spacy
import re
import unidecode
import codecs
import argparse
import json
from markdown import markdown
from bs4 import BeautifulSoup

with open('config/contraction_mapping.json') as f:
        contraction_mapping = json.load(f)

nlp = spacy.load('en_core_web_sm', disable=["parser", "tagger", "ner"])

class Corpus2Vecs(object):
    '''
    fits and transforms the corpus into the form for keras to train with
    '''
    def __init__(self, modelFile = None):
        '''
        initilizes the class
        --------------------
        input:
        modelFile: str or None, the location of the word2vec model to use
        '''
        if modelFile is not None:
            self.model = Word2Vec.load(modelFile)

    def _word2idx(self, word):
        '''
        transforms a word into a dictionary index
        -----------------------------------------
        input: str, the word to be transformed

        return:
        index: int, the index of the word in the dictionary, if word not in dictionary returns 0
        '''
        if word in self.model.wv.vocab:
            return self.model.wv.vocab[word].index
        return 0
    
    def fit(self, X):
        '''
        finds and sets the max length of the inputs
        -------------------------------------------
        input:
        x: list, the corpus
        '''
        self.max_size = int(len(max(X, key=len)) * .25)

    def transform(self, X):
        '''
        takes the corpus and transforms it into indexs for model training
        -----------------------------------------------------------------
        input:
        X: list, the corpus

        return:
        train_X: list, the corpus resized and strings turned into dictionary indexs
        '''
        train_x = np.zeros((len(X), self.max_size), dtype=int)
        for index, i in enumerate(X):
            for j, word in enumerate(i):
                if j < self.max_size:
                    train_x[index,j] = self._word2idx(word)
        return train_x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in Files to make word2vec model')
    parser.add_argument('--data', type=str, default='Amz_book_review_short.parquet', help='location of data')
    args = parser.parse_args()
