# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:49:20 2019

@author: holys
"""
import numpy as np

#a = np.array([0.3,2.9,4.0])
#print(a)
#exp_a=np.exp(a)
#print(exp_a)
#
#sum_exp_a=np.sum(exp_a)
#
#y=exp_a/sum_exp_a
#print(y)

a= np.array([3,4,5])
#np.exp(a)/np.sum(np.exp(a))
c = np.max(a)

def softmax(a):
    c=np.max(a)
    exp_a = np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
print("soft max for array a:",a," is ",softmax(a))
print("double check sum: ",np.sum(softmax(a)))

#MNIST lab!
import sys, os, tensorflow
#sys.path.append(os.pardir)
#import struct

##(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True, normalize=False)
#def load_mnist(path, kind='train'):
#
#    """Load MNIST data from `path`"""
#
#    labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
#
#    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
#
#    with open(labels_path, 'rb') as lbpath:
#
#        magic, n = struct.unpack('>II',lbpath.read(8))
#
#        labels = np.fromfile(lbpath,dtype=np.uint8)
#
#    with open(images_path, 'rb') as imgpath:
#
#        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
#
#        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
#
#    return images, labels
#
#
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from pylab import *

#---------------如果清空了就重新下--------------------------------------------------
#mnist = input_data.read_data_sets(os.path.dirname(__file__)+"\\data",one_hot=True) 

X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels
batch_X, batch_Y = mnist.train.next_batch(64)


print("训练图像：",X_train.shape) 
print("训练标签：",Y_train.shape)
print("测试图像：",X_test.shape) 
print("测试标签：",Y_test.shape) 
#显示一个图像
img = X_train[0]
label = Y_train[0]
print(label)

print(img.shape)
img=img.reshape(28,28)
print(img)
imshow(img)

#def img_show(img):
#    pil_img=Image.fromarray(np.uint8(img))
#    print(pil_img)
##    pil_img.show()
##    imshow(pil_img)
#    
#img_show(img)






