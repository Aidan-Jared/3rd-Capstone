from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)


with open('static/model.pkl') as f:
    model = pickle.load(f)


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
    pred = str(model.predict([data])[0])
    return render_template('form/predict.html', article=data, predicted=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)