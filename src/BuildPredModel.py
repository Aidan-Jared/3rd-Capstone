import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
#import spacy
from imblearn.under_sampling import RandomUnderSampler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.initializers import Constant
# from keras.utils import multi_gpu_model
from MakeWord2Vec import Corpus2Vecs, text_prep
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight 
import h5py
import argparse
import json
import pyarrow.parquet as pq
import s3fs
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
s3 = s3fs.S3FileSystem()

def buildModel(vocab_size, emdedding_size, pretrained_weights):
    '''
    the keras model structure
    -------------------------
    input:
    vocab_size: int, the size of the vocab
    embedding_size: int, the size of the word2vec vectors
    pretrained_weights: numpy array, the weights to use for the embedding layer

    returns:
    model: keras model
    '''
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, embeddings_initializer=Constant(pretrained_weights), trainable=False))
    #model.add(TimeDistributed(Dense(units=emdedding_size, use_bias=False)))
    model.add(LSTM(units=int(emdedding_size / 4), dropout=.5, kernel_initializer='RandomNormal'))
    # model.add(Dense(units=20, activation='tanh'))
    model.add(Dense(units=1, activation='relu', kernel_initializer='RandomNormal'))
    #model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=Adam(lr=.1), loss= 'mse', metrics=["mse"],)
    return model

def split_data(fileName):
    '''
    reads in data and returns the corpus and target of the data
    -----------------------------------------------------------
    input:
    fileName: str, the location of the file in the s3 bucket

    output:
    corpus: list, the cleaned corpus for the model
    y: list, the targets for the model
    '''
    df = pq.ParquetDataset(fileName, filesystem=s3).read_pandas().to_pandas()
    df, discard = train_test_split(df, train_size=.2, stratify=train['star_rating'], random_state=42)
    discard = 0
    corpus, y = text_prep(df)
    return corpus, y

def plot_loss(history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valadation'], loc='upper left')
        plt.savefig('images/model_loss.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in Files to make LSTM model')
    parser.add_argument('--config', type=str, default='config/Config.json', help='location of file to use as config')
    parser.add_argument('--train', type=str, default="s3://capstone-3-data-bucket-aidan/data/train_data.parquet", help='location of file to use for training')
    parser.add_argument('--val', type=str, default="s3://capstone-3-data-bucket-aidan/data/valadation_data.parquet", help='location of file to use for valadation')
    parser.add_argument('--test', type=str, default="s3://capstone-3-data-bucket-aidan/data/test_data.parquet", help='location of file to use for test')
    parser.add_argument('--word2vecModel', type=str, default='models/word2vec.model', help='location of the premade word2vec model')
    args = parser.parse_args()

    with open(args.config) as f:
        config_PM = json.load(f)['PredModel']
    
    print('loading data')
    train, y_train = split_data(args.train)
    val, y_val = split_data(args.val)
    test, y_test = split_data(args.test)
    
    print('formating data')
    vectors = Corpus2Vecs(modelFile=args.word2vecModel)
    vectors.fit(train)
    X_train = vectors.transform(train)
    X_val = vectors.transform(val)
    X_test = vectors.transform(test)
    word_model = Word2Vec.load(args.word2vecModel)

    print('saving transformation')
    with open('models/vector.pkl', 'wb') as f:
        pickle.dump(vectors, f)

    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape
    
    print('starting model training')
    model = buildModel(vocab_size, emdedding_size, pretrained_weights)
    history = model.fit(X_resampled, y_resampled, epochs=config_PM['epoch'], verbose=config_PM['verbose'], batch_size=config_PM['batch_size'], validation_data=(X_val, y_val))
    print(history.history['loss'])
    model.save('models/BookPresentModel.h5')
    
    plot_loss(history)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)