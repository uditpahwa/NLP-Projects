import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import nltk
from nltk.tokenize import  sent_tokenize
from nltk.tokenize import  word_tokenize
from nltk.tokenize import  TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import  stopwords
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from gensim.models import Word2Vec
import re
import pickle
nltk.download('stopwords')
stop_words=stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag

data = pd.read_csv('/Users/uditpahwa/Downloads/nlp-getting-started/train.csv')
####### STEMMING SENTENCES
#### porter stemmer
stemmer = PorterStemmer()
stemmed_text=[]
for i in data['text']:
    words = i.lower().split()
    words = [stemmer.stem(word) for word in words if not word in stop_words]
    words = ' '.join(words)
    stemmed_text.append(words)
######### snowball stemmer
stemmer = SnowballStemmer(language='english')
stemmed_text=[]
for i in data['text']:
    words = i.lower().split()
    words = [stemmer.stem(word) for word in words if not word in stop_words]
    words = ' '.join(words)
    stemmed_text.append(words)
stemmed_text[100]
######### wordnet lemmatizer
######### cleaning up text
all_text=list(data['text'])
for i in range(len(all_text)):
    all_text[i]=re.sub('https[^\n]*','',all_text[i])
    all_text[i]=re.sub('@[\w]*',' ',all_text[i])
    all_text[i] = re.sub('[^a-zA-Z#]', ' ', all_text[i])

stemmer = WordNetLemmatizer()
nltk.download('wordnet')
stemmed_text=[]
for i in all_text:
    words = i.lower().split()
    words = [stemmer.lemmatize(word) for word in words if not word in stop_words]
    words = ' '.join(words)
    stemmed_text.append(words)
stemmed_text[100]
########## word 2 vec
model = Word2Vec(stemmed_text, vector_size=200, window=10, min_count=1)
model.train(stemmed_text, total_examples=len(stemmed_text), epochs=1000)
keys=set(model.wv.index_to_key)
final_embeddings = []
for sentence in stemmed_text:
    sum = np.zeros(200)
    for token in sentence:
        if token in keys:
            sum+=model.wv[token]
        else:
            print(token)
    final_embeddings.append(sum)


######### vectorisation
cv=CountVectorizer(max_features=2500)
cv=TfidfVectorizer(max_features=4000)

###### Use for TFidf or CV
X=cv.fit_transform(stemmed_text).toarray()
###### Use for word2vec
X = np.array(final_embeddings)
y=data['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)

############## Model trained on TF-IDF

############## XGboost
xgb=XGBClassifier(n_estimators=5000,max_depth=10,min_child_weight=6,gamma=1,learning_rate=0.11,
                  colsample_bytree=0.6,colsample_bynode=0.7,reg_lambda =2,reg_alpha=2,early_stopping_rounds=10)
model = xgb.fit(X_train,y_train,eval_set=[(X_test,y_test)])
y_pred_test=model.predict_proba(X_test)[:,1]
y_pred_train=model.predict_proba(X_train)[:,1]
test_score=roc_auc_score(y_test,y_pred_test)
train_score=roc_auc_score(y_train,y_pred_train)
print("Train Score: ",train_score )
print("Test Score: ",test_score )
print("Diff: ",train_score-test_score )

############## Logistic Regression
lr=LogisticRegression(penalty='l2')
model = lr.fit(X_train,y_train)
y_pred_test=model.predict_proba(X_test)[:,1]
y_pred_train=model.predict_proba(X_train)[:,1]
test_score=roc_auc_score(y_test,y_pred_test)
train_score=roc_auc_score(y_train,y_pred_train)
print("Train Score: ",train_score )
print("Test Score: ",test_score )
print("Diff: ",train_score-test_score )
############### finding thresholds
test_results=pd.DataFrame(pd.Series(y_pred_test),columns=['pd'])
test_results['target']=list(y_test)
test_results['pd_bins']=pd.qcut(test_results['pd'],30)
test_results['cnt']=1
test_results.groupby('pd_bins').sum()[['target','cnt']]
pickle.dump(model,open('/Users/uditpahwa/PycharmProjects/NLP_Project/FLASKAPI/model.pkl','wb'))
################ Preparing the output file
classifier=pickle.load(open('/Users/uditpahwa/PycharmProjects/NLP_Project/FLASKAPI/model.pkl','rb'))
test = pd.read_csv('/Users/uditpahwa/Downloads/nlp-getting-started/test.csv')
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
X=data_validation(test)
predictions_test = classifier.predict_proba(X)[:,1]
vectorized_func=np.vectorize(predict_category)
categories_test=vectorized_func(predictions_test)
test['target'] = list(categories_test)
output=test[['id','target']]
output.to_csv('/Users/uditpahwa/PycharmProjects/NLP_Project/final_result.csv')

################# Model trained with word2vec
xgb=XGBClassifier(n_estimators=5000,max_depth=10,min_child_weight=10,gamma=0,learning_rate=0.11,
                  colsample_bytree=1,colsample_bynode=1,reg_lambda =0,reg_alpha=0,early_stopping_rounds=10)
model = xgb.fit(X_train,y_train,eval_set=[(X_test,y_test)])
y_pred_test=model.predict_proba(X_test)[:,1]
y_pred_train=model.predict_proba(X_train)[:,1]
test_score=roc_auc_score(y_test,y_pred_test)
train_score=roc_auc_score(y_train,y_pred_train)
print("Train Score: ",train_score )
print("Test Score: ",test_score )
print("Diff: ",train_score-test_score )




