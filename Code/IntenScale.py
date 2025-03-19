import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# 总的来说，图像像素越亮，对应位置SPN越可信，其中灰度值为252时，权值达到巅峰1
# 像素值>252时认为时过曝光造成的，反而不可信
def IntenScale(X):
    X = X.astype('float32')
    T = 252
    v = 6
    temp = -1 * ((X -T)* (X -T) )/ v
    out = np.exp(temp)
# 打开下面代码注释可以帮助理解这个函数的作用
#     A = np.arange(0, 256, 1)
#     temp2 = -1 * ((A -T)* (A -T) )/ v
#     B = np.exp(temp2)
    # plt.figure(figsize=(3, 3))
    # plt.plot(A,B)
    # plt.show()
    out[np.where(X < T)] = X[np.where(X < T)] / T
    return out