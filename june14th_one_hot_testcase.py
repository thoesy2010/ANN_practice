# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:59:13 2019

@author: holys
"""

from sklearn import preprocessing  
      
enc = preprocessing.OneHotEncoder()  # 创建对象
enc.fit([[0,0,3],[1,1,0],[0,2,1],[1,0,2]])   # 拟合
array = enc.transform([[0,1,3]]).toarray()  # 转化
print(array)