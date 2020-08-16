# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:13:04 2020

@author: nihar
"""

from flask import Flask, request, render_template
import joblib
from keras.models import load_model
import re #regular expression
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
global graph
graph = tf.get_default_graph()
ps = PorterStemmer()


app = Flask(__name__)

#load the model
model=load_model('sentiment analysis.h5')

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    cv=joblib.load('CountVectorizer.pkl')
    x_test =request.form['Text']
    print(x_test)
    review = re.sub('[^a-zA-Z]',' ',x_test)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    review1  = cv.transform([review])
    with graph.as_default():
       prediction1=model.predict([review1])
       print(prediction1)
    if prediction1[0][0]<0.5:
        return render_template('base.html', Prediction='The Tweet is Negative')
    else:
        return render_template('base.html', Prediction='The Tweet is Positive')
        
if __name__ == "__main__":
    app.run(debug=True)

