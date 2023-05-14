from flask import Flask, render_template, request
from nltk.translate.bleu_score import corpus_bleu

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_metric', methods=['POST'])
def calculate_metric():
    metric = request.form['metric']
    candidate = request.form['candidate']
    reference = request.form['reference']

    # Calculate the BLEU score using NLTK
    # Assumes that candidate and reference sentences are already tokenized
    bleu_score = corpus_bleu([[reference]], [candidate])

    return {
        'metric': metric,
        'candidate': candidate,
        'reference': reference,
        'bleu_score': bleu_score
    }
