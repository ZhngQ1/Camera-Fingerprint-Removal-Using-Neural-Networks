import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from playground import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 修改 padding 参数
            nn.ReLU(),  # 添加 ReLU 激活函数
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 修改 padding 参数
            nn.ReLU(),  # 添加 ReLU 激活函数
            nn.MaxPool2d(2, 2),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 修改 padding 参数
            nn.ReLU(),  # 添加 ReLU 激活函数
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 修改 padding 参数
            nn.ReLU(),  # 添加 ReLU 激活函数
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # 修改 padding 参数
            nn.Sigmoid(),  # 添加 Sigmoid 激活函数
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CameraFingerprintRemovalDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        """
        input_dir: 存储原始图片的文件夹路径
        target_dir: 存储去除相机指纹图片的文件夹路径
        transform: 应用于每对图片的转换操作
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])

        base_name = self.filenames[idx].split('_')
        target_file_name = str(int(base_name[0]) + 5) + '_' + base_name[1]
        target_path = os.path.join(self.target_dir, target_file_name)

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# 定义图像预处理
print("*****Define image preprocess*****")
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 实例化数据集
print("*****Instantiate the data set*****")
train_dataset = CameraFingerprintRemovalDataset(
    input_dir="dataset/training/input/1",
    target_dir="dataset/training/target/6",
    transform=transform
)

test_dataset = CameraFingerprintRemovalDataset(
    input_dir="dataset/testing/input/2",
    target_dir="dataset/testing/target/7",
    transform=transform
)

# 创建 DataLoader
print("*****Create a DataLoader*****")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 初始化模型、优化器和损失函数
print("*****Initializes the model, optimizer, and loss function*****")
model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 模型训练
print("*****Model training*****")
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2,"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 50
loss_list = [0.0] * 50

for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    train_loss = 0.0
    for data in train_loader:

        inputs, _ = data
        inputs = inputs.to(device)
        
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, inputs)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        train_loss += loss.item() * inputs.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    
    loss_list.append(train_loss)
    print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}')

# 定义模型保存路径
model_save_path = "Model/model.pth"

# 保存模型状态字典
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# 测试模型
print("*****Model testing*****")
model = Autoencoder()

# 定义模型加载路径
model_load_path = "Model/model.pth"

# 加载模型状态字典
model.load_state_dict(torch.load(model_load_path))

model.eval()  # 设置模型为评估模式
test_loss = 0.0
output_path = "dataset/testing/target/2"
os.makedirs(output_path, exist_ok=True)

with torch.no_grad():  # 关闭梯度计算
    for i, data in enumerate(test_loader):
        inputs, _ = data
        inputs = inputs.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        test_loss += loss.item() * inputs.size(0)
        for j in range(outputs.size(0)):
            vutils.save_image(outputs[j], os.path.join(output_path, f'output_{i * test_loader.batch_size + j}.png'))
    test_loss = test_loss / len(test_loader.dataset)
    
print(f'Test Loss: {test_loss:.4f}')
print(f'输出图片已保存到 {output_path}')