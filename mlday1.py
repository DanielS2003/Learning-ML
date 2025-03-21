#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
day 1 of learning ML, linear regression and k-nearest neighbors
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#time studied per week in minutes
time_studied = np.array([20,50,32,65,23,43,10,5,22,35,29,56]).reshape(-1,1)
#score on a test 
scores = np.array ([56,83,47,93,47,82,45,23,55,67,57,89]).reshape(-1,1)

model = LinearRegression()
model.fit(time_studied,scores)

plt.scatter(time_studied,scores)
plt.plot(np.linspace(0,70,100).reshape(-1,1),model.predict(np.linspace(0,70,100).reshape(-1,1)),'r')
plt.ylim(0,100)
plt.show()

"""
the following code breaks the data into parts and trains 
on one part and then fits to the remaining data
to generate a score to evaluate accuracy
"""
time_train,time_test,score_train,score_test=train_test_split(time_studied,scores,test_size=0.2)

model = LinearRegression()
model.fit(time_train,score_train)

print(model.score(time_test,score_test))
