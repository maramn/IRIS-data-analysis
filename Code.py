# Importing libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes = True)
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('/Users/nagarjuna/PycharmProjects/IRIS_dataanalysis/Iris.csv')
dataset.head(5)

#Removing the S.no column
dataset = dataset.drop('Id', axis = 1)

#Summary of the dataset

print(dataset.shape) # Tells you the dimensions of the dataset
print(dataset.info()) #Tells you about individual
print(dataset.describe()) #Gives you statistical analysis for individual entity
print(dataset.groupby('Species').size()) # Grouping by individual entity

#Visualizations
dataset.plot(kind = 'box', sharex = False, sharey = False) #Boxplot
dataset.boxplot()
dataset.hist()
dataset.boxplot(by = 'Species', figsize=(10,10))
sns.violinplot(data = dataset, x='Species', y='SepalLengthCm') #distribution

from pandas.plotting import scatter_matrix
#Scatter plot matrix
scatter_matrix(dataset, figsize=(10,10))
sns.pairplot(dataset, hue='Species')

#Seperating data into dependant and independant variables

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#Splitting the dataset into training and test dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

#Logistic regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(x_train, y_train)

y_pred =  classifier.predict(x_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import accuracy_score

print('accuracy score is', accuracy_score(y_pred,y_test)) #96

#NAIVE BAYES

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

print('The score is', accuracy_score(y_pred,y_test)) #96

#Support vector machines

from sklearn.svm import SVC
classifier = SVC()

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print('The score is', accuracy_score(y_pred,y_test))

#KNN

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

print('the score is', accuracy_score(y_pred,y_test))
