import numpy as np
import cv2
from NoiseExtractFromIm import NoiseExtractFromIm
from cor import cor
from Zeromean import Zeromean
from WienerInDFT import WienerInDFT
from crosscorr import crosscorr
from PCE import PCE
import os
import csv
import matplotlib.pyplot as plt


#输入一个图片路径，输出噪声残差Noisex和灰阶矩阵imx
def extractFingerprint(path):
    imx = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imx = cv2.resize(imx, (3120, 4160))
    # imx = imx0[0:832, 0:624]
    Noisex = NoiseExtractFromIm(imx,2)
    Noisex = Zeromean(Noisex)
    std = np.std(Noisex)
    Noisex = WienerInDFT(Noisex,std)
    return Noisex,imx

#TODO:请仿照extractFingerprint的实现，从多张参考图片而不是一张当中提取指纹，将多张图片的指纹融合，保存到磁盘备用
def extractFingerprintFromMultipleFiles(path):
    files = os.listdir(path)

    # 初始化为大小相同的0矩阵
    '''
    sample_image_path = os.path.join(path, files[1])
    sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = sample_image.shape
    '''
    Noise_accumulator = np.zeros((832*5, 624*5))

    for file in files:
        # 跳过对标准相机指纹文件的处理
        if file == "merged_noisex_A.npy" or file == "merged_noisex_B.npy":
            continue
        # 获得图片的完整路径，然后得到图片的指纹
        file_path = os.path.join(path, file)
        Noisex, imx = extractFingerprint(file_path)

        # 将相机指纹叠加
        if np.all(Noise_accumulator == 0):
            Noise_accumulator = Noisex
        else:
            Noise_accumulator += Noisex

    # 计算平均的相机指纹
    average_Noisex = Noise_accumulator / len(files)
    # 根据图片来源不同保存为不同的标准相机指纹的文件
    if(path == './dataset/2/PIC/subfolder1'):
        np.save('./extract_FP/test_merged_noisex_2.npy', average_Noisex)
    else:
        np.save('./extract_FP/test_merged_noisex_7.npy', average_Noisex)

# 输入两张图片路径，输出匹配相似度（PCE）
def calculatePCE(path1,path2):
    PRNU,_ = extractFingerprint(path1)
    Noisex,imx = extractFingerprint(path2)
    imx_float = np.array(imx,dtype='float32')
    #计算correlation时需要将实际场景光照与相机指纹乘算后，再和噪声残差进行匹配。因为噪声残差的强度和场景光照是相关的
    C = crosscorr(Noisex,imx_float*PRNU)
    Out = PCE(C)
    res = Out["PCE"]
    print(f"result = {res}")
    return res

#TODO:请仿照calculatePCE的实现，将测试图片与保存到磁盘上的相机指纹进行匹配，输出PCE值
def calculatePCEwithRegisteredImage(path1,path2):
    Noisex, imx = extractFingerprint(path1)
    PRNU = np.load(path2)
    imx_float = np.array(imx,dtype='float32')

    C = crosscorr(Noisex, imx_float * PRNU)
    Out = PCE(C)
    res = Out["PCE"]
    print(f"result = {res}")
    return res


# 提取指定设备的相机指纹保存为文件
print("extract")
'''extractFingerprintFromMultipleFiles("./dataset/2/PIC/subfolder1")
extractFingerprintFromMultipleFiles("./dataset/7/PIC/subfolder1")'''

# 计算reference中图片的PCE作为基准
print("calculate PCE")
test_files = os.listdir("./dataset/2/PIC/subfolder1")
data = []
PCE_A_array = []
PCE_B_array = []
count = 0

for tfile in test_files:
    count += 1
    print(count)
    file_path = os.path.join("./dataset/2/PIC/subfolder1", tfile)

    PCEA = calculatePCEwithRegisteredImage(file_path, './extract_FP/test_merged_noisex_2.npy')
    PCEB = calculatePCEwithRegisteredImage(file_path, './extract_FP/test_merged_noisex_7.npy')
    PCE_A_array.append(PCEA)
    PCE_B_array.append(PCEB)
    
    if(PCEA > PCEB):
        data.append(['test ' f"{tfile}", 'A'])
        # PCE_A_array.append(PCEA)
    else:
        data.append(['test ' f"{tfile}", 'B'])
        # PCE_B_array.append(PCEB)

with open('result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

#TODO：请绘制在与A，B各自参考指纹匹配时PCE分布的直方图，并按照PPT中的格式保存匹配结果。根据直方图选择一个合适的匹配阈值
plt.hist(PCE_A_array, bins=200, alpha=0.5, edgecolor='black', facecolor='red', label='A Class')
plt.hist(PCE_B_array, bins=200, alpha=0.5, edgecolor='black', facecolor='blue', label='B Class')

plt.title('PCE Distribution of Class A')
plt.title('PCE Distribution of Class B')
plt.xlabel('PCE value')
plt.ylabel('Frequency')

plt.legend()
plt.show()   