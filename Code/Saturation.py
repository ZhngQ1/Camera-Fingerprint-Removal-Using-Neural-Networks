import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pywt
# 函数作用：去除过曝光区域，判别条件如下
# 1）图片的最大值>250, 2)像素自己，以及上下左右的值均等于图片最大值，
def Saturation(X):
    X = X.astype('float32')
    SaturMap = np.ones_like(X)
    Xh = (X - np.roll(X,1,axis=0)).astype('bool')
    Xv = (X - np.roll(X,1,axis=1)).astype('bool')
    SaturMap = Xh & Xv & np.roll(Xh,-1,axis=0) & np.roll(Xv,-1,axis=1)
    maxX = X.max()
    if maxX>250:
        left = (X == maxX)
        right = ~SaturMap
        SaturMap = ~(left & right)
    return SaturMap
