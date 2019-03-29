import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gensim.models.word2vec import Word2Vec
import spacy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding
from keras.layers import TimeDistributed
from VectorPipeline import Corpus2Vecs

class Word2Vect(object):
    def __init__(self, nlp, fileName=None):
        self.nlp = nlp
        self.fileName = fileName
    
    def _clean_text(self,X):
        cleaned_text = Corpus2Vecs(nlp, Vectorize=False, modelFile=None).transform(X)
        return cleaned_text
    
    def fit(self, X, min_count = 1, window = 5, epoch = 200, size = 100):
        cleaned_text = self._clean_text(X)
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


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm', disable=["parser", "tagger"])
    table = pq.read_table('Amz_book_review_short.parquet')
    df = table.to_pandas()
    corpus = df['review_body'].values

    W2V = Word2Vect(nlp, fileName='models/testword2vec.model')
    W2V.fit(corpus, size=300)