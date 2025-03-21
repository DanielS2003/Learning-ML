#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
day 3, linear regression again, with pandas
using data from UC Irvine's ML data library
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "absences","failures", "studytime"]]
predict = "G3"

x = np.array(data.drop([predict], axis=1))
y =np.array(data[predict])

best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        #save the best model
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

#load model
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficients: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])
    
#p can be changed to observe different relationships
p="G1"

style.use("ggplot")
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
