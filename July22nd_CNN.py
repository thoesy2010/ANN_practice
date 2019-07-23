# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:57:00 2019

@author: holys
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialize CNN
classifier = Sequential()

#step1 
classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), activation = 'relu',input_shape =(64,64,3) ))#filters:~64 input_shape: 图片大小，颜色
#step2 max pooling 降维度得到特征图
classifier.add(MaxPooling2D(pool_size = (2,2)))

#add hidden layer
classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))



#step3 flattening->可输入！
classifier.add(Flatten())

#step4 full connection
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1 , activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#part 2
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(#!!!!!
    training_set, #!!!!!
    steps_per_epoch=250, #pic_num / batch_size
    epochs=25,
    validation_data=test_set,
    validation_steps=62.5)