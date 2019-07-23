# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:21:43 2019

@author: holys
"""

import numpy as np

x = np.arange(-5, 5, 1)
y = np.arange(-5, 5, 1)
xx, yy = np.meshgrid(x, y, sparse=True)   # 为一维的矩阵
xx1, yy1 = np.meshgrid(x, y )  # 转换成二维的矩阵坐标

#  这样的目的画成一个
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D