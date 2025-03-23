#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
day 4, KNN again, using UC Irvine's car evaluation data set
"""
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clss = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint,door,persons,lug_boot,safety))
y = list(clss)

xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(xtrain,ytrain)
acc = model.score(xtest,ytest)

print(f"Accuracy: {acc}")

predicted = model.predict(xtest)
names = ["unacc", "acc","good","vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data:", tuple(map(int, xtest[x])), "Actual: ", names[ytest[x]])
    
    
