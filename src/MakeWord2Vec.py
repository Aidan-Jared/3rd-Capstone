import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import spacy
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Bidirectional, Embedding
# from keras.layers import TimeDistributed
from VectorPipeline import Corpus2Vecs
import argparse
import json
import pyarrow.parquet as pq
import s3fs
s3 = s3fs.S3FileSystem()

class Word2Vect(object):
    def __init__(self, nlp, fileName=None):
        self.nlp = nlp
        self.fileName = fileName
    
    def fit(self, X, min_count = 1, window = 5, epoch = 200, size = 100, load = None):
        if load is None:
            word_model = Word2Vec(
                X,
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
            word_model.train(X, total_examples=word_model.corpus_count, epochs = epoch)
            if self.fileName != None:
                word_model.save(self.fileName)
            else:
                return word_model

def text_prep(df):
    text = df['review_body_clean']
    y = df['star_rating'].values
    text = [i.tolist() for i in text.values]
    return text, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in Files to make word2vec model')
    parser.add_argument('--config', type=str, default='config/Config.json', help='location of file to use as config')
    parser.add_argument('--train', type=str, default="s3://capstone-3-data-bucket-aidan/data/train_data.parquet", help='location of file to use as corpus')
    parser.add_argument('--modelFile', type=str, default='models/testword2vec.model', help='location of where to save the finished model')
    parser.add_argument('--load', type=str, default = None, help='location of model to load and continue training')
    args = parser.parse_args()
    
    # df = spark.read.parquet(args.corpusFile)
    
    # train, val, test = df.randomSplit([0.7, 0.1, 0.2], seed=427471138)
    train = pq.ParquetDataset(args.train, filesystem=s3).read_pandas().to_pandas()
    
    train, y = text_prep(train)
    
    with open(args.config) as f:
        config_WV = json.load(f)['Word2Vec']

    nlp = spacy.load(config_WV['spacy_model'], disable=config_WV['spacy_disable'])
    
    W2V = Word2Vect(nlp, fileName=args.modelFile)
    W2V.fit(train, min_count=config_WV['min_count'], window=config_WV['window'], epoch=config_WV['epoch'], size=config_WV['size'], load=args.load)