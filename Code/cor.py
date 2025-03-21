import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pywt

def cor(X, Y):
    Xzm = X - np.mean(X)
    Yzm = Y - np.mean(Y)
    XY = np.sum(Xzm*Yzm)
    XX = np.sum(Xzm*Xzm)
    YY = np.sum(Yzm*Yzm)
    p = XY/ (np.power(XX, 0.5) * np.power(YY, 0.5))

    return p
