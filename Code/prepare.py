import os
import shutil

def batch_rename_images(device, folder_path):
    # 获取文件夹中所有文件的列表
    files = os.listdir(folder_path)
    # 初始化文件序号
    file_number = 1
    
    for file in files:
        # 构造原始文件的完整路径
        old_file_path = os.path.join(folder_path, file)
        # 获取文件的扩展名
        extension = os.path.splitext(file)[1]
        # 构造新文件名和完整路径
        new_file_name = f"{device}_{file_number}{extension}"
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # 重命名文件
        shutil.move(old_file_path, new_file_path)
        # 或使用 os.rename(old_file_path, new_file_path) 也可以
        
        # 更新文件序号
        file_number += 1

def batch_copy_images(source_folder, destination_folder):
    # 确保目标文件夹存在，如果不存在，则创建它
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # 获取源文件夹中所有文件的列表
    files = os.listdir(source_folder)
    
    for file in files:
        # 构造源文件的完整路径
        source_file_path = os.path.join(source_folder, file)
        # 构造目标文件的完整路径
        destination_file_path = os.path.join(destination_folder, file)
        
        # 复制文件
        shutil.copy(source_file_path, destination_file_path)



# 调用函数
for i in range(10, 11):
    folder_path = f"dataset/{i}/PIC/subfolder1"
    batch_rename_images(i, folder_path)
    # 调用函数
    source_folder = f"dataset/{i}/PIC/subfolder1"
    destination_folder = f"dataset/training/target"
    batch_copy_images(source_folder, destination_folder)