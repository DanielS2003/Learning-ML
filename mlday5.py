#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml day 5, introduction to neural networks
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ["t-shirt/top", "trouser", "pullover", "dress", "coat", 
               "sandal", "shirt", "sneaker", "bag", "ankle boot"]

train_images = train_images/255
test_images = test_images/255

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax")
    ])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Accuracy: ", test_acc)