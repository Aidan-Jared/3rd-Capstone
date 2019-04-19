import numpy as np
import pandas as pd
import spacy
import argparse
import json
from bs4 import BeautifulSoup
import re
import unidecode
import json

# Spark starting information
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=org.apache.hadoop:hadoop-aws:2.7.3 pyspark-shell"
import pyspark as ps
spark = (ps.sql.SparkSession.builder 
        .master("local[9]") 
        .appName("capstone") 
        .config("spark.executor.memory", '7g') 
        .config("spark.driver.memory", "6g")
        .config("spark.memory.offHeap.enabled",True)
        .config("spark.memory.offHeap.size","7g")
        .getOrCreate()
        )
sc=spark.sparkContext
sc.setLogLevel("OFF")
hadoop_conf=sc._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3n.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
hadoop_conf.set("fs.s3n.awsAccessKeyId", '')
hadoop_conf.set("fs.s3n.awsSecretAccessKey", '')

from pyspark.sql import functions as sf
from pyspark.sql.types import StringType

with open('config/contraction_mapping.json') as f:
        contraction_mapping = json.load(f)

def text_cleaner(Doc):
        '''
        reformats, tokenizes, removes stop words, and lemmanizes the inputed text
        -------------------------------------------------------------------------
        input:
        Doc: str, the document to be cleaned

        return:
        Doc: str, the cleaned and processed document
        '''
        if Doc is not None:
            Doc = ' '.join(BeautifulSoup(Doc).findAll(text=True))
            try:
                Doc = unidecode.unidecode(codecs.decode(Doc, 'unicode_escape'))
            except:
                Doc = unidecode.unidecode(Doc)
            Doc = re.sub("’", "'", Doc)
            Doc = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in Doc.split(" ")])
            Doc = nlp(Doc)
            final_tokens = []
            for t in Doc:
                if t.is_space or t.like_num or t.like_url or t.is_stop:
                    pass
                else:
                    if t.lemma_ == '-PRON-':
                        final_tokens.append(t.text)
                    else:
                        sc_removed = re.sub("[^a-zA-Z]", '', t.lemma_)
                        if len(sc_removed) > 1:
                            final_tokens.append(sc_removed)
            return ' '.join(final_tokens)
        else:
            return Doc

@sf.pandas_udf("string", sf.PandasUDFType.SCALAR)
def textclean(x):
    '''
    adds text_cleaner to spark functions
    '''
    return x.apply(text_cleaner)
textclean = spark.udf.register('textclean', textclean)


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm', disable=["parser", "tagger", "ner"])
    parser = argparse.ArgumentParser(description='data to clean and split')
    parser.add_argument('--data', type=str, default='Amz_book_review_short.parquet', help='location of data')
    args = parser.parse_args()
    
    print('starting data import \n')
    df = spark.read.parquet(args.data)

    print('starting text cleaning \n')
    df = df.select(['product_id', 'product_title', 'review_headline', 'review_body', 'star_rating'])
    
    df = df.withColumn('joined', sf.concat(sf.col('review_headline'), sf.lit(' - '), sf.col('review_body')))
    df = df.withColumn('review_body_clean', sf.split(textclean(df.joined), ' '))
    df = df.select(['product_id', 'product_title','review_body_clean', 'star_rating'])
    df.printSchema()
    
    print('starting train test split \n')
    train, val, test = df.randomSplit([0.025, 0.0025, 0.005], seed=427471138)
    
    print('exporting train \n')
    train.write.parquet("s3n://capstone-3-data-bucket-aidan/data/train_data.parquet",mode="overwrite")
    train.unpersist()
    print('exporting val \n')
    val.write.parquet("s3n://capstone-3-data-bucket-aidan/data/valadation_data.parquet",mode="overwrite")
    val.unpersist()
    print('exporting test \n')
    test.write.parquet("s3n://capstone-3-data-bucket-aidan/data/test_data.parquet",mode="overwrite")
    test.unpersist()