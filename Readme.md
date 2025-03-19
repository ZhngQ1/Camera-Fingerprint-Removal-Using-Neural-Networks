# Readme

- Code 中的文件较多，文件结构如下：

  ```
  ├── dataset
  ├── extract_FP
  │   ├── merged_noisex_1.npy
  │   ├── merged_noisex_6.npy
  │   ├── test_merged_noisex_2.npy
  │   └── test_merged_noisex_7.npy
  ├── Model
  │   └── model.pth
  ├── output
  │   ├── output_0
  │   ├── ...
  │   └── output_99
  ├── PCE_result
  │   ├── Figure 1
  │   ├── Figure 2
  │   └── Figure 3
  ├── autoencoder.py
  ├── cor.py
  ├── crosscorr.py
  ├── FPcreateColor.py
  ├── FPcreate.py
  ├── IntenScale.py
  ├── NoiseExtractFromIm.py
  ├── NoiseExtract.py
  ├── PCE.py
  ├── playground.py
  ├── prepare.py
  ├── result.csv
  ├── Saturation.py
  ├── WienerInDFT.py
  └── Zeromean.py
  ```

- 其中 **dataset** 包含训练和测试的图片，以序号 **1-10** 区分设备（由于太大就不上传了）

- **extract_FP** 包含从某一个设备拍摄的图片中提取的指纹

- **Model** 是训练好的模型文件

- **output** 是设备 **2** 输入模型得到的去除指纹的图片

- **PCE_result** 是进行测试的时候绘制的直方图图片

- 另外就是源代码文件，主要文件是 **autoencoder.py**，其余都是 **lab1** 的文件，仅在 **playground.py** 中做了修改

