# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pylab import *
import cv2 as cv
# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
figure()
# 显示原图
My_img = Image.open("DDD.jpg")
#print(My_img.mode,My_img.size.My_img.format)
subplot(231)
title(u'原图', fontproperties=font)
axis('off')
imshow(My_img)
#imshow(My_img)
#My_img.convert('L')
#imshow(My_img.convert('L'))
#My_img.thumbnail((128,128))
#imshow(smallimg)
#subplot(231)
'''box = (50,50,300,300)
region = My_img.crop(box)
imshow(region)
'''
#显示灰度图
gray()
Limg=My_img.convert('L')
subplot(232)
title(u'灰度图', fontproperties=font)
axis('off')
imshow(Limg)

#复制并粘贴区域
subplot(233)
dox = (50,50,150,150)
region = My_img.crop(dox)
Newregion = region.transpose(Image.ROTATE_180)
My_img.paste(Newregion,dox)
title(u'复制贴图',fontproperties=font)
axis('off')
imshow(My_img)

#略缩图
subplot(234)
My_img = Image.open("DDD.jpg")
size = 128,128
My_img.thumbnail(size)
title(u'缩略图', fontproperties=font)
axis('off')
imshow(My_img)

#调整图像尺寸
subplot(235)
My_img = Image.open("DDD.jpg")
size = 64,128
My_img = My_img.resize(size)
title(u'缩放图',fontproperties = font)
axis('off')
imshow(My_img)

#旋转图像45度
My_img = Image.open("DDD.jpg")
subplot(236)
My_img = My_img.rotate(45)
title(u'旋转45度',fontproperties=font)
axis('off')
imshow(My_img)

show()





