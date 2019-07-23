# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:31:54 2019

@author: hu
"""

from PIL import Image
from pylab import *

# 添加中文字体支持
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

im = array(Image.open("DDD.jpg"))
subplot(121)
#imshow(im)
x=[100,50,300,150]
y=[200,300,100,250]

# 使用红色星状标记绘制点
plot(x,y,'r*')

#绘制连接两个点的线（默认为蓝色）
#plot(x[:2],y[:2])
plot(x[:2],y[:2])
plot((x[2],x[3]),(y[2],y[3]),'w')
title(u'绘制empire.jpg', fontproperties=font)

imshow(im)

im2 = array(Image.open("DDD.jpg").convert('L'))
gray()
subplot(122)
imshow(im2)

figure()
subplot(223)
contour(im2,origin='image')
title(u'轮廓图像图',fontproperties=font)



    