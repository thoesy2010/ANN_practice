# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:41:21 2019

@author: hu
"""
'''
import numpy as np
x = np.array([0,1])
w = np.array([0.5,0.5])
b = - 0.7
print(w*x)
'''

from PIL import Image
from numpy import *
from pylab import *

im = array(Image.open('DDD.jpg').convert('L'))
figure()
gray()
subplot(231)
imshow(im)

rev_im=255-im#对图像进行反向处理
domain_im=int(im2.min()),int(im2.max()) #将图像像素值变换到100...200区间
subplot(232)
title('当前值域%d',fontproperties=font)
imshow(rev_im)




im3=(100/255)*im +100
subplot(233)
title(u'试试',fontproperties=font)
imshow(im3)

im4=(im/255)**2*255
subplot(234)
imshow(im4)#对像素值求平方后得到的图像
##---------------------------------------------------------------------------------
#直方图均衡化
import imtools

def histeq(im,nbr_bins=256):
    imhist,bins = histogram(im.flatten(),nbr_nbins,normed = True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf/cdf[-1]
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf


im2, cdf = imtools.histeq(im)
subplot(235)
hist(im2.flatten(),128,normed=true)

show()

