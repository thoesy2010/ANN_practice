# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:26:05 2019

@author: holys
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
##
###------------------------
#def sigmoid(x):
#    y = 1/(1+np.exp(-x))
#    return y
##
#def derivative_sigmoid(y):
#    return y*(1-y)
##forward loop
# 
#alphaList = [0.001,0.01,0.1,1,10,100,1000,5000]
#
#layer0=np.array([[0,1,0],[0,0,1],[1,0,0]])#输入值
#
##确保每次的随机数一样
#
##y = np.array([[1],[0],[0]])#真实数据
#
#for alpha in alphaList:
#    print ("\nTraining With alpha value:" ,alpha)
#    np.random.seed(1)
#    weight1 = np.random.random((3,4))-1
#    weight2 = np.random.random((4,1))-1
#        
#    for i in range(60000):
#        
#        #forward 
#        layer1 = sigmoid(np.dot(layer0,weight1))#z1
#        layer2 = sigmoid(np.dot(layer1,weight2)) #z2
#        #if(i%10000==0):
#        #    print("layer2 is :",layer2)
#        layer_2_error = y - layer2 # y-z2 = z2_error
#        if(i%10000==0):
#            print("layer_2_error is: ",np.mean(np.abs(layer_2_error)))
#        #backward propa
#        #L2的损失函数求导
#        layer_2_delta = layer_2_error*derivative_sigmoid(layer2) # dL2 = z2_error* dz2
#        
#        weight2_derivative = np.dot(layer1.T,layer_2_delta) # dw2 = dL2 * z1
#        
#        #updateweight2
#        weight2 -= alpha*weight2_derivative #new_w2 = old_w2 - learning_rate*dw2_error 
#        #    if(i%10000==0):
#        #        print("weight2 is: ", weight2)
#        #L1
#        layer_1_error = layer_2_delta.dot(weight2.T) #layer1error = dL2*w1    #逻辑可能有毛病
#        #if(i%10000==0):
#        #    print("layer_1_error is: ",np.mean(np.abs(layer_1_error)))
#        
#        
#        layer_1_delta = layer_1_error*derivative_sigmoid(layer1)#dz1_error = z1_error*dz1
#        weight1_derivative = np.dot(layer_1_delta.T,layer_2_delta) #a1*dz1*da2
#        
#        #updateweight1
#        weight1 -= alpha*weight1_derivative.T # w - a*dw
#        #    if(i%10000==0):
#            #        print("weight1 is: ", weight1)
##------------------------