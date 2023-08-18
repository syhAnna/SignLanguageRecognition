import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# import the dataset
dataset_train = pickle.load(open('/Users/yuneshou/Desktop/2022winter/CS219/sign-language-recognition/ik_annotated/kaggle_asl.pkl', 'rb'))
dataset_test =  pickle.load(open('/Users/yuneshou/Desktop/2022winter/CS219/sign-language-recognition/ik_annotated/kaggle_asl_test.pkl', 'rb'))
x_train = dataset_train.iloc[:, :len(dataset_train.columns) - 1].values
y_train = []
for c in dataset_train.iloc[:, -1].values:
    y_train.append(ord(c))

x_test = dataset_test.iloc[:, :len(dataset_test.columns) - 1].values
y_test = []
for c in dataset_test.iloc[:, -1].values:
    y_test.append(ord(c))

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# print(y_test)
# print(y_pred)

# Making the Confusion Matrix
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)

print(ac)