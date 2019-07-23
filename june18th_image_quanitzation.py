# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:42:20 2019

"""

from PIL import Image
import numpy as np
from pylab import *

im = array(Image.open('DDD.jpg').convert('L'))
figure()
gray()
subplot(231)
imshow(im)
(row,col)=im.shape
new = np.zeros((row,col))

im2 = im
GreyNum = 4
r= 256/GreyNum #区间大小
r2 = 255/(GreyNum-1)

for j in range (row):
    for m in range(col):
        for i in range(GreyNum):
            if(im[j,m] < i*r):
                new[j,m] = 0 +(i-1)*r2
            
subplot(232)
imshow(new)