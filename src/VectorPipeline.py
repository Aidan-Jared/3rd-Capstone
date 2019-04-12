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

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }

nlp = spacy.load('en_core_web_md', disable=["parser", "tagger", "ner"])

class Corpus2Vecs(object):
    def __init__(self, modelFile = None):
        if modelFile is not None:
            self.model = Word2Vec.load(modelFile)

    # def text_cleaner(self, Doc):
    #     html = markdown(Doc)
    #     Doc = ' '.join(BeautifulSoup(html, features="html.parser").findAll(text=True))
    #     try:
    #         decoded = unidecode.unidecode(codecs.decode(Doc, 'unicode_escape'))
    #     except:
    #         decoded = unidecode.unidecode(Doc)
    #     apostrophe_handled = re.sub("’", "'", decoded)
    #     expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    #     Doc = self.nlp(expanded)
    #     final_tokens = []
    #     for t in Doc:
    #         if t.is_punct or t.is_space or t.like_num or t.like_url or t.is_stop:
    #             pass
    #         else:
    #             if t.lemma_ == '-PRON-':
    #                 final_tokens.append(str(t).lower())
    #             else:
    #                 sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_).lower())
    #                 if len(sc_removed) > 1:
    #                     final_tokens.append(str(t.lemma_).lower())
    #     return ' '.join(final_tokens)

    def _word2idx(self, word):
        if word in self.model.wv.vocab:
            return self.model.wv.vocab[word].index
        return 0
    
    # def clean_text(self, X):
    #     return self._text_cleaner(X)

    def fit(self, X):
        # cleaned = self.clean_text(X)
        self.max_size = len(max(X, key=len))

    def transform(self, X):
        #  cleaned = self.clean_text(X)
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
    
    # df = spark.read.parquet(args.data)
    # df = df.select(['product_id', 'product_title', 'review_headline', 'review_body', 'star_rating'])
    
    # nlp = spacy.load('en_core_web_md', disable=["parser", "tagger", "ner"])
    
    # df = df.withColumn('joined', sf.concat(sf.col('review_headline'), sf.lit(' - '), sf.col('review_body')))
    
    # f = udf(Corpus2Vecs(nlp, modelFile=None).clean_text, StringType())
    # df = df.withColumn('review_body_clean', f(df.joined))
    
    # df = df.withColumn('review_body_clean', sf.split('review_body_clean', ' '))
    
    # df = df.select(['product_id', 'product_title','review_body_clean', 'star_rating'])
    # df.show(5)
    
    # df.write.parquet("s3n://capstone-3-data-bucket-aidan/data/cleaned_data.parquet",mode="overwrite")
