# Importing the required libraries
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Using a count vectoriser to create the required bag of words
cv = CountVectorizer(max_features=5000,ngram_range=(1,3)) 
#making .pkl file for count vectoriser for preprocessing the input text in our web app
pickle.dump(cv, open('transform.pkl','wb'))

# Processing the training data
# Import the dataset
fake_dataset = pd.read_csv("/dataset/train.csv")

#Preprocess the dataset to remove NAN values & set index
fake_dataset.dropna(inplace = True)
fake_dataset.reset_index(inplace=True)

#Applying stemming and removing the stop words
#Here we will be using news content to create bag of words
ps = PorterStemmer()
corpus = []
for i in range(0,len(fake_dataset)):
    review = re.sub('az-A-Z',' ',fake_dataset['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Separating independent and dependent variables
# X--> denotes independent variables
X = cv.fit_transform(corpus).toarray()
# Y--> dependent, to be predicted (fake or not fake)
Y = fake_dataset.iloc[:, -1].values

#Processing the testing dataset
test_dataset = pd.read_csv("/dataset/test.csv")
test_dataset.dropna(inplace = True)
test_dataset.reset_index(inplace=True)

#Applying stemming and removing the stop words
ps = PorterStemmer()
corpus = []
for i in range(0,len(test_dataset)):
    review = re.sub('az-A-Z',' ',test_dataset['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Taking the answer dataset for accuracy prediction
submit_dataset = pd.read_csv("/dataset/submit.csv")

#Taking only the non null values for the answer database
temp = []
j=0
for i in range(0,len(submit_dataset)):
    if (submit_dataset['id'][i]==test_dataset['id'][j]):
        j=j+1
        temp.append(submit_dataset['label'][i])

#Test cases
X_test = cv.fit_transform(corpus).toarray()
Y_test = np.array(temp)

#Applying the classifiers

#1.) Decision Tree Classifier
model1 = DecisionTreeClassifier()
model1.fit(X, Y)
Y_pred1 = model1.predict(X_test)
ac = accuracy_score(Y_test,Y_pred1)
# Accuracy found : 0.817418649165983

#2 Random Forest classifier
model2 = RandomForestClassifier(n_estimators=10)
model2.fit(X, Y)
Y_pred2 = model2.predict(X_test)
ac2 = accuracy_score(Y_test,Y_pred2)
# Accuracy found : 0.8264424391577796

#3 Naive bayes classifier
model3 = GaussianNB()
model3.fit(X, Y)
Y_pred3 = model3.predict(X_test)
ac3 = accuracy_score(Y_test,Y_pred3)
# Accuracy found : 0.7358490566037735

#We found out the best accuracy result for random forest
#So we will use this model to implement our web app
#Dumping our decision tree classifier model to pkl file which will be later used by our webapp
pickle.dump(model2, open('nlp_model.pkl','wb'))

#Model is ready to be used for prediction
