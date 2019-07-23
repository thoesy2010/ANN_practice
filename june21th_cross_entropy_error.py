# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:35:06 2019

@author: hu
"""
import numpy as np
def cross_entropy_error(y,t):
    delta = 1e-7
    ans = -np.sum(t*np.log(y+delta))
    return ans

t=[0,0,1]
y=[0.1,0.5,0.4]
ans=cross_entropy_error(np.array(y),np.array(t))
print(ans)


#mini batch

def cross_entropy_error_minibatch(y,t):
    if(y.ndim==1):
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size = y.shape[0]    
    
    delta = 1e-7
    ans = -np.sum(t*np.log(y[np.arange(batch_size),t]+delta))
    return ans

t=[0,0,1]
y=[0.1,0.5,0.4]
ans=cross_entropy_error_minibatch(np.array(y),np.array(t))
print(ans)

#数值微分
def function_1(x):
    return 0.01*x**2+0.1*x
import matplotlib.pylab as plt


def numerical_diff(f,x):
    h = 1e-4 #0.0001
#    print(f(x+h),f(x-h))
    return (f(x+h)-f(x-h))/(2*h)

x = np.arange(0.0,20.0,0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()
numerical_diff(function_1,5)



    