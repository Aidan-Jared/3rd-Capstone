import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import spacy
# from imblearn.under_sampling import RandomUnderSampler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.initializers import Constant
# from keras.utils import multi_gpu_model
from MakeWord2Vec import Corpus2Vecs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight 
import h5py
import argparse
import json
import pyarrow.parquet as pq
import s3fs
import pickle
s3 = s3fs.S3FileSystem()

def text_prep(df):
    text = df['review_body_clean']
    y = df['star_rating'].values
    text = [i.tolist() if i is not None else [''] for i in text]
    return text, y

def buildModel(vocab_size, emdedding_size, pretrained_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, embeddings_initializer=Constant(pretrained_weights), trainable=False))
    #model.add(TimeDistributed(Dense(units=emdedding_size, use_bias=False)))
    model.add(LSTM(units=emdedding_size, dropout=.5))
    model.add(Dense(units=int(emdedding_size / 4), activation='tanh'))
    model.add(Dense(units=1, activation='relu'))
    #model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=Adam(lr=.001), loss= 'mse', metrics=["mse"],)
    return model

def Rounding(y):
    y = np.round(y)
    y[y > 5] = 5
    y[y < 1] = 1
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in Files to make LSTM model')
    parser.add_argument('--config', type=str, default='config/Config.json', help='location of file to use as config')
    parser.add_argument('--train', type=str, default="s3://capstone-3-data-bucket-aidan/data/train_data.parquet", help='location of file to use for training')
    parser.add_argument('--val', type=str, default="s3://capstone-3-data-bucket-aidan/data/valadation_data.parquet", help='location of file to use for valadation')
    parser.add_argument('--test', type=str, default="s3://capstone-3-data-bucket-aidan/data/test_data.parquet", help='location of file to use for test')
    parser.add_argument('--word2vecModel', type=str, default='models/word2vec.model', help='location of the premade word2vec model')
    # parser.add_argument('--load', type=str, default = None, help='location of model to load and continue training')
    args = parser.parse_args()

    with open(args.config) as f:
        config_PM = json.load(f)['PredModel']
    
    train = pq.ParquetDataset(args.train, filesystem=s3).read_pandas().to_pandas()
    val = pq.ParquetDataset(args.val, filesystem=s3).read_pandas().to_pandas()
    test = pq.ParquetDataset(args.test, filesystem=s3).read_pandas().to_pandas()
    
    train, y_train = text_prep(train)
    val, y_val = text_prep(val)
    test, y_test = text_prep(test)
    

    nlp = spacy.load(config_PM['spacy_model'], disable=config_PM['spacy_disable'])
    vectors = Corpus2Vecs(modelFile=args.word2vecModel)
    vectors.fit(train)
    X_train = vectors.transform(train)
    X_val = vectors.transform(val)
    X_test = vectors.transform(test)
    word_model = Word2Vec.load(args.word2vecModel)
    with open('models/vectortransform.pkl', 'wb') as f:
        pickle.dump(vectors, f)

    sample_weight = compute_sample_weight('balanced', y_train)

    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    # X_smt, y_smt = Smote(X_train, y_train)

    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape

    model = buildModel(vocab_size, emdedding_size, pretrained_weights)
    history = model.fit(X_train, y_train, epochs=config_PM['epoch'], batch_size=config_PM['batch_size'], verbose=config_PM['verbose'], sample_weight=sample_weight)
    print(history.history['loss'])
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(1,-1)
    y_pred = Rounding(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)
    model.save(config_PM['model_name'])