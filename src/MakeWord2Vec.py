import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gensim.models.word2vec import Word2Vec
import spacy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding
from keras.layers import TimeDistributed
from VectorPipeline import Corpus2Vecs
import argparse
import json

class Word2Vect(object):
    def __init__(self, nlp, fileName=None):
        self.nlp = nlp
        self.fileName = fileName
    
    def _clean_text(self,X):
        cleaned_text = Corpus2Vecs(nlp, modelFile=None).clean_text(X)
        return cleaned_text
    
    def fit(self, X, min_count = 1, window = 5, epoch = 200, size = 100, load = None):
        cleaned_text = self._clean_text(X)
        if load is None:
            word_model = Word2Vec(
                cleaned_text,
                min_count=min_count,
                window=window,
                iter=epoch,
                size=size)
            if self.fileName != None:
                word_model.save(self.fileName)
            else:
                return word_model
        else:
            word_model = Word2Vec.load(load)
            word_model.train(cleaned_text, total_examples=word_model.corpus_count, epochs = epoch)
            if self.fileName != None:
                word_model.save(self.fileName)
            else:
                return word_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in Files to make word2vec model')
    parser.add_argument('--config', type=str, default='config/Config.json', help='location of file to use as config')
    parser.add_argument('--corpusFile', type=str, default='Amz_book_review_short.parquet', help='location of file to use as corpus')
    parser.add_argument('--modelFile', type=str, default='models/testword2vec.model', help='location of where to save the finished model')
    parser.add_argument('--load', type=str, default = None, help='location of model to load and continue training')
    args = parser.parse_args()

    with open(args.config) as f:
        config_WV = json.load(f)['Word2Vec']

    nlp = spacy.load(config_WV['spacy_model'], disable=config_WV['spacy_disable'])
    table = pq.read_table(args.corpusFile)
    df = table.to_pandas()
    corpus = df[config_WV['corpus_col']].values

    W2V = Word2Vect(nlp, fileName=args.modelFile)
    W2V.fit(corpus, min_count=config_WV['min_count'], window=config_WV['window'], epoch=config_WV['epoch'], size=config_WV['size'], load=args.load)