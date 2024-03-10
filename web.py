import streamlit as st
from flask import Flask, Response, request, jsonify, render_template
import pickle
from flask_cors import CORS
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize
from pythainlp import word_vector
import string

app = Flask(__name__)
# CORS(app,methods=['GET','POST'])

model = word_vector.WordVector(model_name="thai2fit_wv").get_model()  # load thai2fit_wv from pythainlp

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

load_model = pickle.load(open('LR_Embedding.pkl', 'rb'))

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

def aggregate_embeddings(embeddings):
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return None

def fake_news(news):
    input_data = [news]
    embed_news = []

    for document in input_data:
        tokens = word_tokenize(document)
        embeddings = get_word_embeddings(tokens)
        document_embedding = aggregate_embeddings(embeddings)
        embed_news.append(document_embedding)
        prediction = load_model.predict(embed_news)
    return prediction

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/api/predict', methods=['GET','POST'])
def predict():
    print(request.method)
    if request.method == 'POST':
        data = request.get_json()
        news_content = data['news']

        if not news_content:
            return jsonify({'error': 'Please enter some text for analysis'})
        
        if contains_only_numbers(news_content):
            return jsonify({'error': 'Input should not contain only numbers'})
        
        if contains_only_special_characters(news_content):
            return jsonify({'error': 'Input should not contain only special characters'})

        prediction_class = fake_news(news_content)

        response_json = jsonify({'prediction': prediction_class[0]})
        print("Response from /predict:", response_json)

        return response_json
    
    else:
        return Response(status=200)

    

if __name__ == '__main__':
    st.title('Thai News Headlines Analysis ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "", height=200)
    show_warning = False

    predict_btt = st.button("Check")
    
    if predict_btt:
    # Validate input
        if not sentence:
            show_warning = True
        elif contains_only_numbers(sentence):
            st.warning('Input should not contain only numbers')
        elif contains_only_special_characters(sentence):
            st.warning('Input should not contain only special characters')
        else:
            # Make prediction
            prediction_class = fake_news(sentence)
            print(prediction_class)
            if prediction_class == ['ham']:
                st.success('Reliable')
            elif prediction_class == ['spam']:
                st.warning('Unreliable')

# Display warning if needed
if show_warning:
    st.warning('Please enter some text for analysis')