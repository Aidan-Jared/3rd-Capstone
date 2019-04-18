# Review Scores From Writen 

## Motavation and Project Explanation

For this project I decided to use NLP and RNN's to predict a user given review score from the users writen review. In order to do this I used the Amazon review dataset and used the Book product category to build and test the model. This data set was made up of a lot of user metadata and product data but I decided to only look at the Review Body, Review Headline, and Star Rating columns to build the model.

For EDA please refer to [this other project of mine](https://github.com/Aidan-Jared/NLP-Data-Featurization) but for this project I was dealing with heavily imbalanced data.

## Project Overview

- Read Data in Through Spark
- Clean Text Through Spacy
- Build Word2Vec on training set
- Use Word2Vec embeddings and LSTM to predict user scores
- Develop Flask App

## Text Cleaning

In order to do this project I needed to read in the data through Spark and s3 then clean the text in Spark with user defined functions and save the resulting dataframes to s3 for latter access. In order to acomplish this I tried using an AWS EMR but ran into problems when bootstraping the instances so I ended up using a docker image on an EC2 instance and chaning Sparks executor memory to prevent memory overflow errors.

The function I wrote to clean the text used Beautiful Soup and unidecode to remove formatting and Spacy with the en_core_web_sm model as the tokenizer. The documents ended up being a combination of the review headline and review body which were the two text columns of the dataset

```python
def text_cleaner(Doc):
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
```

This code takes each document removes any markdown formatting and turns all unicode into ascii then replaces all contractions with the full expantion. The resulting string is then read into spaCy where it tokenized. I then run through the tokens and remove whitespace, numbers, urls, and stop words from the document and then return lemmanized words with cases perserved. Then using a premade spark function I split the strings into lists for the word2vec and predictive models down the line. This results in:

```
This book was just right! - Just the book that I needed to get me through before the new season of Downton Abbey premieres.  I really enjoyed this engaging story. My only complaint is that I read it too fast!  Jo Baker does such a good job of bringing thesedownstairs people to life, especially given, that I am re-reading Pride and Prejudice and I don't hear a peep from these characters within those pages.  Rest assured there is, indeed, a wholeworld going on below the stairs.  An important thing to note is that if you've never read Prideand Prejudice, this story can still be enjoyed.  It stands on its own.  This is one of my favorites for 2013.
```

becomes:

```
book right book need new season Downton Abbey premiere enjoy engage story complaint read fast Jo Baker good job bring downstairs people life especially give read Pride Prejudice hear peep character page Rest assure world go stair important thing note read Pride Prejudice story enjoy stand favorite
```

I then applied a random split to the Spark dataframe and got a train, valadation, and testing set to work with for the rest of the project. With this done I exported these dataframes to an s3 bucket for latter use.

## Word2Vec

For data featurization I decided to do a Word2Vec model due to an understanding of the process and familiarity with Gensim. Word2Vec is an unsurpervized learning technique of embedding a word into a dense float vector where the vector contains context about the word. The model is build in what is called the skip-gram method where a context window moves across the string and tries to predict the word in the middel of the window. After this process is compleate, the calculated weights become the Word2Vec vector.

After some testing and checking I found that these settings:

```
min_count: 15 #Amount of times a word needs to apear to be part of the model
window: 4 #The context window that is used to predict each word
epoch: 300
size: 300 #The ending size of the vectors
``` 

worked the best and produced the best embeddings as you can see here:

|Original Word|Most Simular|Second Most Simular|Third Most Simular|Forth Most Simular|Fifth Most Simular|
|--------|------|------|------|-----|------|
|fiction|Nationally|Blackoaks|appreciators|Perspectives|TSNOTD|
|fantasy|manipulatively|Palacios|Oregan|precisly|Auel|
|romance|jb|Absorb|Aishling|deem|Institutionalizing|
|religion|frei|Hadoke|Cariou|malapropisms|conceited|
|history|Promoters|cockamamie|aspiration|dustbin|Undercurrents|

## Predictive Model

To build my predictive model I decided to look at sentiment analysis Neuaral Nets as a base because review scores can be seen as a type of sentiment. In order to apply to Word2Vec model the first layer of my model was an embedding layer to which I added in the Word2Vec weights and I then made that layer untrainable. This layer then went into a LSTM and then a dense layerwith a tanh acvtivation fuction into a singluar dense output with a relu activation. I then rounded the results and found the MSE of the model.

In order to make the model work I looked at the training data and found the largest string of the data and set that to be the lenght of all my vectors so that I could capture all the context of each sentance. After this my model replaced all strings with an integer index that corisponded with a dictionary entry and then fed the data into my model.

My model ended up


```
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 300)         12360900
_________________________________________________________________
lstm_1 (LSTM)                (None, 75)                112800
_________________________________________________________________
dense_1 (Dense)              (None, 20)                1520
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 21
=================================================================
Total params: 12,475,241
Trainable params: 114,341
Non-trainable params: 12,360,900
_________________________________________________________________
```

### training and val loss