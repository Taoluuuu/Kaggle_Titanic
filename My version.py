# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:07:28 2017

@author: taolu
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('fivethirtyeight')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('/Users/taolu/Documents/ML/Titanic/train.csv')
test_df = pd.read_csv('/Users/taolu/Documents/ML/Titanic/test.csv')

train_df.describe()
train_df.describe(include=['O'])
train_df.isnull().sum()
#Age and Cabin have a lot of missing values
train_df['Cabin2'] = train_df.Cabin.str.extract(r'(\w)\d+')
train_df['Cabin2'].replace(np.NaN, 'Z', inplace = True)


train_df[['Survived','Sex']].groupby(['Sex'],as_index=False).mean().sort_values(\
        by='Survived',ascending=False)

#train_df.groupby(['Survived','Sex'])['Survived'].count()

#Relationship between sex and survive rate
g = sns.FacetGrid(train_df, row='Sex', col = 'Survived')
g.map(plt.hist, 'Age')
g.add_legend()  
#filling the age NAs into 5 bins
train_df['Age'].isnull().sum()/train_df['Age'].size
train_df.loc[train_df.Age.isnull(),"Age"] = train_df['Age'].dropna().median()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean()
train_df.loc[train_df.Age <= 16.336,'Age'] = 1
train_df.loc[(train_df.Age > 16.336) & (train_df.Age <= 32.252),'Age'] = 2
train_df.loc[(train_df.Age > 32.252) & (train_df.Age <= 48.168),'Age'] = 3
train_df.loc[(train_df.Age > 48.168) & (train_df.Age <= 64.084),'Age'] = 4
train_df.loc[(train_df.Age > 64.084) & (train_df.Age <= 80),'Age'] = 5
train_df.Age.unique()

#Relationship between Pclass and survive rate
train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).count()
#
#train_df[['Embarked','Survived']].groupby(['Embarked'])[''].count().to_frame()
train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).count()
train_df.groupby(['Pclass','Sex','Survived'])['Survived'].mean()
g = sns.FacetGrid(train_df, row='Embarked')
g.map(sns.pointplot, 'Pclass','Survived','Sex',palette='deep')
g.add_legend()

train_df[['Embarked', 'Pclass', 'Sex', 'Survived']].groupby(['Embarked', 'Pclass', 'Sex'], as_index=False).mean().sort_values(by=['Embarked', 'Pclass', 'Sex'], ascending=True)

#Relationship between SibSp and Survived
pd.crosstab(train_df.SibSp, train_df.Survived)
sns.pointplot('SibSp','Survived','Sex',data = train_df,palette='deep')

train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.pointplot('Parch','Survived','Sex',data = train_df,palette='deep')

#Adding title column
train_df['Title'] = train_df.Name.str.extract(r'(\w+)\.')
pd.crosstab(train_df['Title'], train_df['Survived'])
train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)
train_df['Title'].replace('Mlle', 'Miss', inplace = True)
train_df['Title'].replace('Ms', 'Miss', inplace = True)
train_df['Title'].replace('Mme', 'Mrs', inplace = True)
train_df['Title'].unique()
sns.pointplot('Title','Survived',data = train_df,palette='deep')
sns.factorplot('Age','Survived',col='Title',data = train_df)
#train_df[["Title", "Survived"]].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
title_mapping = {"Mr" : 1, "Miss" : 2, "Mrs" : 3, "Master" : 4, "Rare" : 5}
train_df['Title'] = train_df['Title'].map(title_mapping)
train_df.drop(['Name','PassengerId'], axis = 1, inplace = True)
train_df.head(5)

train_df['Cabin2'] = train_df.Cabin.str.extract(r'(\w)\d+')
train_df['Cabin2'].replace(np.nan, 'Z', inplace = True)

train_df['Sex'] = train_df['Sex'].map({"male" : 1, "female" : 0})

train_df['Embarked'] = train_df['Embarked'].map({"S" : 1, "C" : 2 , "Q" : 3})
train_df.loc[train_df.Embarked.isnull(),'Embarked'] = train_df['Embarked'].dropna().median()
train_df.columns
train_df.drop(['AgeBand','Cabin2','Ticket'], axis = 1, inplace = True)
train_df.drop(['Cabin'], axis = 1, inplace = True)

#take a look at the fare column
g = sns.FacetGrid(train_df, row = 'Survived')
g.map(sns.barplot, 'Sex','Fare', ci = None)
g.add_legend()

train_df.loc[train_df.Fare.isnull(), 'Fare'] = train_df['Fare'].dropna().median()
train_df['FareBand'] = pd.cut(train_df['Fare'], 5)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index = False).count()
train_df.loc[train_df.Fare <= 102.466,'Fare2'] = 1
train_df.loc[(train_df.Fare > 102.466) & (train_df.Fare <= 204.932),'Fare2'] = 2
train_df.loc[(train_df.Fare > 204.932) & (train_df.Fare <= 307.398),'Fare2'] = 3
train_df.loc[(train_df.Fare > 307.398) & (train_df.Fare <= 409.863),'Fare2'] = 4
train_df.loc[(train_df.Fare > 409.863) & (train_df.Fare < 513.329),'Fare2'] = 5
train_df[['Fare2', 'Survived']].groupby(['Fare2'], as_index = False).count()
train_df.drop(['FareBand'], axis = 1, inplace = True)
train_df.drop(['Fare'], axis = 1, inplace = True)

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

X_test = test_df


#fit models to the data
#Logistical regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
X_train.head(5)
log_accuracy = logreg.score(X_train, Y_train)
coeff_df = pd.DataFrame(train_df.columns.delete(0))

train_df.columns.delete(0)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)

#KNN model
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
#Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
#Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# Decision Tree Classifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
#Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
#Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

