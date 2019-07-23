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

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
figure()

My_img = Image.open("DDD.jpg")
#imshow(My_img)
#My_img.convert('L')
#imshow(My_img.convert('L'))
#My_img.thumbnail((128,128))
#imshow(smallimg)
#subplot(231)
imshow(My_img)
box = (50,50,300,300)
region = My_img.crop(box)
#imshow(region)