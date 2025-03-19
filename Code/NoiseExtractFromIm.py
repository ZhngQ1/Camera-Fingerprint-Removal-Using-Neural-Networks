import numpy as np
import cv2
import matplotlib.pyplot as plt

from NoiseExtract import NoiseExtract



def NoiseExtractFromIm(image,sigma):
    # 4层小波变换
    L = 4
    height, width = image.shape[0], image.shape[1]
    # 对应'db4'的小波核函数
    qmf = [ .230377813309,.714846570553,.630880767930, -0.027983769417,
        -0.187034811719, .030841381836,.032883011667, -0.010597401785]
    # 核心代码，用小波去噪来提取SPN，后期可换成其他去噪算法
    Noise = NoiseExtract(image, qmf, sigma, L)
    #plt.figure(figsize=(3, 3))
    #plt.imshow(Noise, cmap="gray", vmin=-10, vmax=10)
    #plt.show()
    return Noise







