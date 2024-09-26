import os

import numpy as np

from flask import Flask, request,render_template, json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

app = Flask(__name__)

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/predict',methods=["Get","POST"])
def predict():
    uploaded_file = request.files['file']
    df = pd.read_csv(uploaded_file)
    X = data_validation(df)
    with open("model.pkl", 'rb') as file:
            classifier = joblib.load(file)
    predictions_test = classifier.predict_proba(X)[:,1]
    vectorized_func=np.vectorize(predict_category)
    categories_test=vectorized_func(predictions_test)
    df.reset_index(inplace=True)
    df['Predictions'] = categories_test
    result=df[['index','Predictions']]
    return result.to_json(orient="split")

def predict_category(i):
    if i>0.409:
        i=1
    else:
        i=0
    return(i)
def data_validation(df):
    stemmer = WordNetLemmatizer()
    stemmed_text = []
    all_text = list(df['text'])
    stop_words = stopwords.words('english')
    for i in range(len(all_text)):
        all_text[i] = re.sub('https[^\n]*', '', all_text[i])
        all_text[i] = re.sub('@[\w]*', ' ', all_text[i])
        all_text[i] = re.sub('[^a-zA-Z#]', ' ', all_text[i])
    for i in all_text:
        words = i.lower().split()
        words = [stemmer.lemmatize(word) for word in words if not word in stop_words]
        words = ' '.join(words)
        stemmed_text.append(words)
    cv = TfidfVectorizer(max_features=4000)
    X = cv.fit_transform(stemmed_text).toarray()
    return X

if __name__ == '__main__':
     app.run(debug=True, port=5002)

