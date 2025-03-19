import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from NoiseExtract import WaveNoise

# 这个函数很重要，可用来去除SPN中同型号相机的共同部分，这些共同部分可能会让同型号不同相机的照片难以区分
# 思想为：这些共同成分在图片上具有很强的周期性，对应在傅里叶变换域上为一个个离散的亮点，所以
# 1）把SPN变换到Fourier 域； 2）用WaveNoise 函数将这些亮点削弱； 3）将削弱后的Fourier系数反变换回
def WienerInDFT(X,sigma):
    M,N = X.shape[0],X.shape[1]
    F = np.fft.fft2(X)
    Fmag = np.abs(F / np.sqrt(M * N))

    # plt.imshow(Fmag,cmap="gray",vmin=-10, vmax=10)
    # plt.colorbar()
    # plt.show()

    NoiseVar = np.power(sigma,2)
    Fmag1 = WaveNoise(Fmag, NoiseVar)

    fzero = np.where(Fmag == 0)
    Fmag[fzero] = 1
    Fmag1[fzero] = 0
    F = F* Fmag1/ Fmag
    NoiseClean = np.real(np.fft.ifft2(F))
    # plt.imshow(NoiseClean,cmap="gray",vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.show()

    return NoiseClean.astype('float32')
