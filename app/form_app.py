from flask import Flask, render_template, request, jsonify
import spacy
import pickle
import json
from bs4 import BeautifulSoup
import re
import unidecode
from src.MakeWord2Vec import Word2Vect
from keras.models import load_model
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

with open('config/contraction_mapping.json') as f:
    contraction_mapping = json.load(f)
with open('models/vectortransform.pkl', 'rb') as f:
    vectors = pickle.load(f)

model = load_model('models/TestModel.h5')

def text_cleaner(Doc):
        if Doc is not None:
            Doc = ' '.join(BeautifulSoup(Doc, "lxml").findAll(text=True))
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
            return final_tokens
        else:
            return Doc

def vectorizor(text):
    tokens = text_cleaner(text)
    tokens = vectors.transform(tokens)
    return tokens


@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('form/index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('form/submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    data = str(request.form['article_body'])
    tokens = vectorizor(data)
    pred = model.predict(tokens)
    return render_template('form/predict.html', article=data, predicted=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
