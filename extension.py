import streamlit as st
from flask import Flask, Response, request, jsonify, render_template
import pickle
from flask_cors import CORS
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pythainlp.tokenize import word_tokenize
from pythainlp import word_vector
import string
from typing import Any
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from th_preprocessor.preprocess import preprocess

app = Flask(__name__)
CORS(app)

model = word_vector.WordVector(model_name="thai2fit_wv").get_model()  # load thai2fit_wv from pythainlp

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()
thvo = r"าิีึืฺุูๅๆ็ํ๊๋ํ๎๏๐๑๒๓๔๕๖๗๘๙"

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = "".join(nopunc)
    
    return [word for word in word_tokenize(nopunc) if word and not re.search(pattern=r"\s+", string=word)]

def dummy(obj: Any) -> Any:
    return obj

def tokenize(text: str, min_char: int = 2, remove_placeholder: bool = False) -> List[str]:
    tokens = word_tokenize(text, keep_whitespace=False)
    
    if remove_placeholder:
        tokens = [token for token in tokens if not token.startswith("WS")]
    
    return [token.strip() for token in tokens if len(token.strip()) >= min_char]


class ThaiPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def preprocess(self, text: str) -> str:
        return preprocess(text)  # call the same preprocess function as previous example

    def transform(self, X) -> pd.Series:
        return pd.Series(X).apply(self.preprocess)


class ThaiTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_placeholders: bool = False, min_char: bool = 2):
        self.remove_placeholders = remove_placeholders
        self.min_char = min_char

    def fit(self, X, y=None, **fit_params):
        return self

    def tokenize(self, text):
        tokens = tokenize(
            text,
            min_char=self.min_char,
            remove_placeholder=self.remove_placeholders,
        )

        return tokens

    def transform(self, X) -> pd.Series:
        return pd.Series(X).apply(self.tokenize)

load_model = pickle.load(open('LinearSVM.pkl', 'rb'))

def starts_with_thai_vowel(text):
    # Regular expression pattern to match sentences starting with Thai vowels
    pattern = r'^([' + thvo + '])'
    
    # Check if the sentence matches the pattern
    return bool(re.match(pattern, text))

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)

    punc = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''

    for con in punc:
        if con in string.punctuation:
            clean_text = clean_text.replace(con, "")
        return clean_text

    con = clean_text

    return con

def contains_only_numbers(text):
    return all(char.isdigit() or char.isspace() for char in text)

def contains_only_special_characters(text):
    return all(char in string.punctuation or char.isspace() for char in text)

def get_word_embeddings(tokens):
    embeddings = [model[word] for word in tokens if word in model]
    return embeddings

# def aggregate_embeddings(embeddings):
#     if embeddings:
#         return sum(embeddings) / len(embeddings)
#     else:
#         return None

# def fake_news(news):
#     input_data = [news]
#     embed_news = []

#     for document in input_data:
#         tokens = word_tokenize(document)
#         embeddings = get_word_embeddings(tokens)
#         document_embedding = aggregate_embeddings(embeddings)
#         embed_news.append(document_embedding)
#         prediction = load_model.predict(embed_news)
#     return prediction

def fake_news(news):
    TfidfVectorizer()
    TfidfTransformer()
    word_tokenize(news, engine="newmm")
    # return (load_model.predict(news))
    prediction = load_model.predict(news)
    return prediction

@app.route('/api/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        news_content = data['news']
        # print("hellwoorld",news_content)

        if not news_content:
            return jsonify({'error': 'Please enter some text for analysis'})
        
        if any(char.isascii() and char.isalpha() for char in news_content):
            return jsonify({'error': 'Rejected: The text should be Thai alphabet characters.'})
        
        if contains_only_numbers(news_content):
            return jsonify({'error': 'Input should not contain only numbers'})
        
        if contains_only_special_characters(news_content):
            return jsonify({'error': 'Input should not contain only special characters'})
        
        if starts_with_thai_vowel(news_content):
             return jsonify({'error': 'Sentence should not start with Thaivowels'})
        else:
            prediction_class = fake_news(news_content)
            response_json = jsonify({'prediction': prediction_class[0]})
            print("Response from /predict:", response_json)
            return response_json
    
    else:
        return Response(status=200)

if __name__ == '__main__':
    app.run()