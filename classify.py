# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

import numpy as np

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    n = len(train_set[0])  # number of features
    W = np.zeros(n)
    b = 0
    finish = True
    for it in range(max_iter):
        for idx in range(len(train_set)):
            x = train_set[idx, :]
            y = train_labels[idx] * 2 - 1  # label in [-1, 1]
            score = np.dot(W, x) + b
            f = np.sign(score)
            if f != y:
                W += learning_rate * y * x
                b += learning_rate * y
                finish = False
        if finish:
            break
    # return the trained weight and bias parameters
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    scores = np.dot(dev_set, W) + b
    dev_labels = [0 if i < 0 else 1 for i in scores]
    # Train perceptron model and return predicted labels of development set
    return dev_labels

def sigmoid(x):
    # return output of sigmoid function given input x
    return 1 / (1 + np.exp(-x))

def trainLR(train_set, train_labels, learning_rate, max_iter):
    n = len(train_set[0])  # number of features
    W = np.zeros(n)
    b = 0
    for it in range(max_iter):
        m = len(train_set)  # number of images
        score = np.dot(train_set, W) + b
        f = sigmoid(score)
        # loss = -1 / m * sum(y * log(f) + (1 - y) * log(1 - f))
        gradient_w = -1 / m * np.dot(np.transpose(train_set), train_labels - f)
        gradient_b = -1 / m * np.sum(train_labels - f)
        if not np.any(gradient_w) and gradient_b == 0:
            break
        W -= learning_rate * gradient_w
        b -= learning_rate * gradient_b
    # return the trained weight and bias parameters
    return W, b

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    W, b = trainLR(train_set, train_labels, learning_rate, max_iter)
    scores = np.dot(dev_set, W) + b
    fs = sigmoid(scores)
    dev_labels = [0 if i < 0.5 else 1 for i in fs]
    # Train LR model and return predicted labels of development set
    return dev_labels

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    return []
