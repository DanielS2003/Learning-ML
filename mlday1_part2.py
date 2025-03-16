#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
day 1 of learning ML, linear regression and k-nearest neighbors
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

print(data.feature_names)
print(data.target_names)

xtrain,xtest,ytrain,ytest=train_test_split(np.array(data.data),np.array(data.target),test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(xtrain,ytrain)

print(clf.score(xtest,ytest))

clf.predict([])