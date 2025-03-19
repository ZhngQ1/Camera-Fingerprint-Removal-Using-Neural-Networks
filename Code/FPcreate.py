# 这是Basic方法，直接用多张突破SPN的平均值作为相机PRNU，简单易理解
# 但为了与同行代码一致，我们不使用这个算法！
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat
from NoiseExtractFromIm import NoiseExtractFromIm
from Zeromean import Zeromean

for i in range(1, 4):
    imx = cv2.imread('Images\P'+str(i)+ '.jpg', cv2.IMREAD_GRAYSCALE)
    Noisex = NoiseExtractFromIm(imx, 2)
    plt.figure(figsize=(3, 3))
    plt.imshow(imx, cmap="gray", vmin=0, vmax=255)
    plt.show()
    if i==1:
        PRNU = Noisex
    else:
        PRNU += Noisex

PRNU = PRNU/(i)
PRNU = Zeromean(PRNU)

mdict = {'PRNU': PRNU}
savemat('mat/PRNU.mat', mdict)
    # data = loadmat('mat/px.mat')
i = 6

