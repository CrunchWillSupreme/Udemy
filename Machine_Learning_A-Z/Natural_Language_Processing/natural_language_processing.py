# Natural Language Processing
import os
os.chdir(r'C:\Users\willh\Udemy\Machine Learning A-Z\P14-Machine-Learning-AZ-Template-Folder\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing')

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', sep = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    #dataset['Review'][0]
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# we will use Classification - Naive Bayes for this example, but typically for NLP use Naive Bayes, Decision Tree, or Random Forest
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split --deprecated
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.20, random_state = 0)

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Accuracy = # predicted correct / total # of predictions
(55+91)/200 # 73%

# Precision = total # predictions / (true positives + false negatives)

# Recall = true positives / (false positives + false negatives)

# F1 score = 2 * Precision * Recall / (Precision + Recall)
