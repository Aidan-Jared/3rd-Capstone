import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gensim.models.word2vec import Word2Vec
import spacy
from imblearn.under_sampling import RandomUnderSampler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.initializers import Constant
from MakeWord2Vec import Corpus2Vecs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight 
import h5py
import argparse
import json

def buildModel(vocab_size, emdedding_size, pretrained_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, embeddings_initializer=Constant(pretrained_weights), trainable=False))
    model.add(TimeDistributed(Dense(units=emdedding_size, use_bias=False)))
    model.add(LSTM(units=emdedding_size, dropout=.5))
    model.add(Dense(units=int(emdedding_size / 4), activation='tanh'))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=Adam(lr=.001), loss= 'mse', metrics=["mse"],)
    return model

def Smote(X, y, random_state = 42, k_neighbors=3, plot = False):
      smt = SMOTE(random_state=42, k_neighbors=3)
      X_smt, y_smt = smt.fit_sample(X, y)
    #   if plot:
    #     df_temp = pd.DataFrame(y_smt, columns=['star_rating'])
    #     df_temp = df_temp['star_rating'].value_counts()
    #     ax = df_temp.plot(kind='bar')
    #     ax.set_xlabel('Catigories', fontsize=12)
    #     ax.set_ylabel('Count', fontsize=12)
    #     ax.set_title("SMOTED Class Distributions", fontsize=16)
    #     plt.savefig('images/SMOTE_class_distributions.png')
      return X_smt, y_smt

def Rounding(y):
    y = np.round(y)
    y[y > 5] = 5
    y[y < 1] = 1
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in Files to make LSTM model')
    parser.add_argument('--config', type=str, default='config/Config.json', help='location of file to use as config')
    parser.add_argument('--TrainFile', type=str, default='Amz_book_review_short.parquet', help='location of file to use as corpus')
    parser.add_argument('--word2vecModel', type=str, default='models/testword2vec.model', help='location of the premade word2vec model')
    # parser.add_argument('--load', type=str, default = None, help='location of model to load and continue training')
    args = parser.parse_args()

    with open(args.config) as f:
        config_PM = json.load(f)['PredModel']
    
    table = pq.read_table(args.TrainFile)
    df = table.to_pandas()

    corpus = df[config_PM['corpus_col']].values
    y = df[config_PM['y_col']].values
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=.2, stratify=y)

    nlp = spacy.load(config_PM['spacy_model'], disable=config_PM['spacy_disable'])
    vectors = Corpus2Vecs(nlp, modelFile=args.word2vecModel)
    vectors.fit(X_train)
    X_train = vectors.transform(X_train)
    X_test = vectors.transform(X_test)
    word_model = Word2Vec.load(args.word2vecModel)

    #class_weight = [{0:1,1:10}, {0:1,1:10},{0:1, 1:8},{0:1, 1:8},{0:1,1:1}]
    sample_weight = compute_sample_weight('balanced', y_train)

    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    # X_smt, y_smt = Smote(X_train, y_train)

    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape

    model = buildModel(vocab_size, emdedding_size, pretrained_weights)
    model.fit(X_train, y_train, epochs=config_PM['epoch'], verbose=config_PM['verbose'], validation_data=(X_test, y_test), sample_weight=sample_weight)
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(1,-1)
    y_pred = Rounding(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)
    # model.save('TestModel.h5')