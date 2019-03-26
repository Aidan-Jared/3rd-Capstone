import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gensim.models.word2vec import Word2Vec
import spacy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.initializers import Constant
from MakeWord2Vec import Corpus2Vecs
from sklearn.model_selection import train_test_split
import h5py

def buildModel(vocab_size, emdedding_size, pretrained_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, embeddings_initializer=Constant(pretrained_weights), trainable=False))
    model.add(LSTM(units=emdedding_size))
    model.add(Dense(units=vocab_size))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=Adam(lr=.001), loss= 'mse', metrics=["mse"])
    return model

if __name__ == "__main__":
    table = pq.read_table('Amz_book_review_short.parquet')
    df = table.to_pandas()
    corpus = df['review_body'].values
    y = df['star_rating'].values
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=.2, stratify=y)

    nlp = spacy.load('en_core_web_sm', disable=["parser", "tagger"])
    vectors = Corpus2Vecs(nlp, Vectorize=True, modelFile='models/testword2vec.model')
    X_train, word_model = vectors.transform(X_train, max_size=50)
    X_test, word_model1 = vectors.transform(X_test, max_size=50)

    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape

    model = buildModel(vocab_size, emdedding_size, pretrained_weights)
    model.fit(X_train, y_train, epochs=1, verbose=1)
    # model.save('TestModel.h5')