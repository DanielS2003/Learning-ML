#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
day 2, support vector machines, decision trees, and random 
forest classification
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

x = data.data
y = data.target

xtrain,xtest,ytrain,ytest=train_test_split(x,y, test_size=0.2)

clf = SVC(kernel='linear', C=3)
clf.fit(xtrain,ytrain)

clf2=KNeighborsClassifier(n_neighbors=3)
clf2.fit(xtrain,ytrain)

clf3=DecisionTreeClassifier()
clf3.fit(xtrain,ytrain)

clf4=RandomForestClassifier()
clf4.fit(xtrain,ytrain)

print(f'SVC: {clf.score(xtest,ytest)}')
print(f'KNN: {clf2.score(xtest,ytest)}')
print(f'DTC: {clf3.score(xtest,ytest)}')
print(f'RFC: {clf4.score(xtest,ytest)}')