#In[]
#Importing the required libraries
from flask import Flask,render_template,url_for,request
import nltk
import pickle
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
 
stemmer = PorterStemmer()
# Load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)
#Home page
@app.route('/')
def home():
    return render_template('home.html')
#Predicting the result
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        temp = []
        data = message
        review = re.sub('[^a-zA-Z]', ' ', data) 
        review = review.lower()
        review = review.split()
        review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        temp.append(review)
        vect = cv.transform(temp).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
# %%
