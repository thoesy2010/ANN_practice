# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:47:13 2019

using adagrad + 2layer  

@author: hu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,[1,2,3]].values
y = dataset.iloc[:,[4]].values
#X['sex_dummy']=X.Gender.map({'Female':0,'Male':1})
#F=pd.get_dummies(X,columns=['Gender'],drop_first=True)

#ms



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
##
#
#
#
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

alphas = [0.001,0.01,0.1,1,10,100,1000]

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
    
#X = np.array([[0,0,1],
#            [0,1,1],
#            [1,0,1],
#            [1,1,1]])
#                
#y = np.array([[0],
#			[1],
#			[1],
#			[0]])

for alpha in alphas:
    print ("\nTraining With Alpha:" + str(alpha))
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((3,6)) - 1
    synapse_1 = 2*np.random.random((6,6)) - 1
    synapse_2 = 2*np.random.random((6,1)) - 1

    for j in range(60000):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        layer_3 = sigmoid(np.dot(layer_2,synapse_2))

        # how much did we miss the target value?
        layer_3_error = layer_3 - y

        if (j% 10000) == 0:
            print ("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_3_error))))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_3_delta = layer_3_error*sigmoid_output_to_derivative(layer_3)
        
        layer_2_error = layer_3_delta.dot(synapse_2.T)
        
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_2 -= alpha * (layer_2.T.dot(layer_3_delta))
        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))

