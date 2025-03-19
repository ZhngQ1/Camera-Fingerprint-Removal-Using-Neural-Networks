# 用图像灰度加权平均的SPN作为 相机PRNU，这也是目前主流做法，也叫MLE方法
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat

from NoiseExtract import NoiseExtract
from Zeromean import Zeromean
from Saturation import Saturation
from IntenScale import IntenScale

# 生成PRNU时sigma为3，而提取单张图片SPN时sigma为2
sigma = 3
L = 4
qmf = [.230377813309, .714846570553, .630880767930, -0.027983769417,
       -0.187034811719, .030841381836, .032883011667, -0.010597401785]
for i in range(1, 4):
    imx = cv2.imread('Images\P'+str(i)+ '.jpg', cv2.IMREAD_COLOR)
    if i == 1:
        # Be careful 只能初始化一次
        PRNU = np.zeros_like(imx).astype('float32')
        NN = np.zeros_like(imx).astype('float32')
    for c in range(0, 3):
        # cv2 颜色通道顺序是B, G, R
        imxSingleChan = imx[:,:,2-c]
        # 对每个颜色通道分别提取SPN生成PRNU，后面会把它们转换回一个通道
        Noisex = NoiseExtract(imxSingleChan,qmf,sigma,L)
        Inten = np.ones_like(Noisex)
        # 计算加权值，如果注释掉这句话则退化到Basic方法，基本思想是亮的区域比暗的区域提取出的SPN更可信
        # 但过曝光区域（一整片区域全是一个大于250的值）提取出的SPN又完全没意义
        Inten = IntenScale(imxSingleChan)*Saturation(imxSingleChan)
        if i == 1:
            PRNU[:,:,c] = Noisex*Inten
            NN[:, :, c] = Inten*Inten
        else:
            PRNU[:,:,c] += Noisex*Inten
            NN[:, :, c] += Inten * Inten

for c in range(0, 3):
    # 分母+1是为了避免除数为0
    PRNU[:, :, c] = PRNU[:, :, c] / (NN[:, :, c]+1)
# Convert RGB-like real data to gray-like output.
PRNU = PRNU[:,:,0]*0.2989+PRNU[:,:,1]*0.5870+PRNU[:,:,2]*0.1140
#plt.figure(figsize=(3, 3))
#plt.imshow(PRNU, cmap="gray", vmin=-5, vmax=5)
#plt.show()

PRNU = Zeromean(PRNU)

# 把它存起来，以后就不用再生成了
mdict = {'PRNU': PRNU}
savemat('mat/PRNUC.mat', mdict)
    # data = loadmat('mat/px.mat')
i = 6

