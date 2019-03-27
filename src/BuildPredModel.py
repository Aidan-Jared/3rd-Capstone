import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gensim.models.word2vec import Word2Vec
import spacy
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding, Dropout
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.initializers import Constant
from MakeWord2Vec import Corpus2Vecs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import h5py

def buildModel(vocab_size, emdedding_size, pretrained_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, embeddings_initializer=Constant(pretrained_weights), trainable=False))
    # model.add(TimeDistributed(Dense(units=emdedding_size)))
    model.add(Bidirectional(LSTM(units=emdedding_size)))
    model.add(Dropout(.5))
    #model.add(Dense(units=vocab_size))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=Adam(lr=.001), loss= 'mse', metrics=["mse"])
    return model

def Smote(X, y, random_state = 42, k_neighbors=3, plot = False):
      smt = SMOTE(random_state=42, k_neighbors=3)
      X_smt, y_smt = smt.fit_sample(X, y)
      if plot:
        df_temp = pd.DataFrame(y_smt, columns=['star_rating'])
        df_temp = df_temp['star_rating'].value_counts()
        ax = df_temp.plot(kind='bar')
        ax.set_xlabel('Catigories', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title("SMOTED Class Distributions", fontsize=16)
        plt.savefig('images/SMOTE_class_distributions.png')
      return X_smt, y_smt

if __name__ == "__main__":
    table = pq.read_table('Amz_book_review_short.parquet')
    df = table.to_pandas()
    corpus = df['review_body'].values
    y = df['star_rating'].values
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=.2, stratify=y)

    nlp = spacy.load('en_core_web_sm', disable=["parser", "tagger"])
    vectors = Corpus2Vecs(nlp, Vectorize=True, modelFile='models/testword2vec.model')
    X_train, word_model = vectors.transform(X_train, max_size=200)
    X_test, word_model1 = vectors.transform(X_test, max_size=200)

    X_smt, y_smt = Smote(X_train, y_train)

    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape

    model = buildModel(vocab_size, emdedding_size, pretrained_weights)
    model.fit(X_smt, y_smt, epochs=10, verbose=1, batch_size=100)
    y_pred = model.predict(X_test)
    y_pred.reshape(1,-1)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)
    # model.save('TestModel.h5')