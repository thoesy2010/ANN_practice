# -*- coding: utf-8 -*-


import numpy as np

#MNIST lab!
import sys, os, tensorflow

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from pylab import *

#---------------如果清空了就重新下--------------------------------------------------
#mnist = input_data.read_data_sets(os.path.dirname(__file__)+"\\data",one_hot=True) 



#global X_train, Y_train, X_test, Y_test

mnist = input_data.read_data_sets(os.path.dirname(__file__)+"\\data",one_hot=True) 
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels 
    #batch_X, batch_Y = mnist.train.next_batch(64)

print("训练图像：",X_train.shape) 
print("训练标签：",Y_train.shape)
print("测试图像：",X_test.shape) 
print("测试标签：",Y_test.shape) 
#显示一个图像
img = X_train[0]
label = Y_train[0]

#print(label)
#print(img.shape)
img=img.reshape(28,28)
#print(img)
imshow(img)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test



def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x.W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3)+b3
    z3 = sigmoid(a3)
    y = softmax(a3)
    
    return y

#x,t = get_data()
#network = init_network()
#
#accuracy_cnt = 0
#for i in range(len(x)):
#    y = predict(network,x[i])
#    p = np.argmax(y)
#    if p == t[i]:
#        accuracy_cnt +=1
#print("acurracy:"+str(float(accuracy_cnt)/len(x)))
    
    





