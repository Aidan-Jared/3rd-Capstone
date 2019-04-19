import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import spacy
from VectorPipeline import Corpus2Vecs
import argparse
import json
import pyarrow.parquet as pq
import s3fs
s3 = s3fs.S3FileSystem()

class Word2Vect(object):
    '''
    Class to train a Word2Vec model and save the model or return the model
    '''
    def __init__(self, fileName=None):
        '''
        initilizes the class
        --------------------
        input:
        fileName: str or None, the path to where the model will be saved, if none
        the model will be returned and not saved
        '''
        self.fileName = fileName
    
    def fit(self, X, min_count = 1, window = 5, epoch = 200, size = 100, load = None):
        '''
        trains the word2vec model from the imputed data
        -----------------------------------------------
        inputs:
        x: list, the data in lists of lists format for the model to train on
        min_count: int, times a word has to apear before being considered
        window: int, the size of the context window
        epoch: int, how many times all the training vectors are used
        size: int, the size of the final vector
        load: str, load a pretrained model and train some more (untested)

        return:
        if no filename is given will return trained model
        '''
        if load is None:
            word_model = Word2Vec(
                X,
                min_count=min_count,
                window=window,
                iter=epoch,
                size=size,
		workers = -1)
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
    '''
    seperates corpus and target and formats corpus for trainging
    ------------------------------------------------------------
    input
    df: dataframe, dataframe of corpus and targes

    returns:
    text: list, corpus formated for training
    y: list, the models targets
    '''
    text = df['review_body_clean']
    y = df['star_rating'].values
    text = [i.tolist() if i is not None else [''] for i in text]
    return text, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in Files to make word2vec model')
    parser.add_argument('--config', type=str, default='config/Config.json', help='location of file to use as config')
    parser.add_argument('--train', type=str, default="s3://capstone-3-data-bucket-aidan/data/train_data.parquet", help='location of file to use as corpus')
    parser.add_argument('--modelFile', type=str, default='models/word2vec.model', help='location of where to save the finished model')
    parser.add_argument('--load', type=str, default = None, help='location of model to load and continue training')
    args = parser.parse_args()

    print('Reading in Data')
    train = pq.ParquetDataset(args.train, filesystem=s3).read_pandas().to_pandas()
    
    train, y = text_prep(train)
    
    with open(args.config) as f:
        config_WV = json.load(f)['Word2Vec']
    
    print('Starting Word2Vec training')
    W2V = Word2Vect(fileName=args.modelFile)
    W2V.fit(train, min_count=config_WV['min_count'], window=config_WV['window'], epoch=config_WV['epoch'], size=config_WV['size'], load=args.load)
    print('finished training', '\n')

    word_model = Word2Vec.load('models/word2vec.model')
    test_lst = ['fiction', 'fantasy', 'romance', 'religion', 'history']
    for i in test_lst:
        print(i)
        print(word_model.wv.most_similar_cosmul(positive=i, topn=5), '\n')

