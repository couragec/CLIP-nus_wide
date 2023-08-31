'''
@Project ：CLIPx
@File    ：nuswide_cwy.py
@IDE     ：PyCharm
@Author  ：Berryc
@Date    ：2023/8/18 19:48
'''

import argparse
import os
import pickle
import random
import PIL
import h5py
import torch
import numpy as np
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip
from sklearn.metrics import accuracy_score, f1_score
from torchvision.transforms import InterpolationMode
import warnings
from torchvision.transforms.transforms import InterpolationMode
# import torch
# torch.cuda.memory_summary(device=None, abbreviated=False)

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message="floor_divide is deprecated")
warnings.filterwarnings("ignore", message="Argument interpolation should be of type InterpolationMode instead of int.", category=UserWarning)


# 加载CLIP模型和令牌   函数用于从给定的文件中加载字典数据。在这里，它被用来加载图像名称的字典
def load_dict(filename_):
    with open(filename_, 'rb') as f:         #'rb' 表示以二进制只读模式打开文件
        ret_dict = pickle.load(f)
    return ret_dict

class ValDataset(Dataset): #是一个数据集类，需要实现 __len__ 和 __getitem__ 方法。

    def __init__(self, args, image_names, transforms):
        self.src = args.data_path
        train_loc = os.path.join(self.src, 'features', 'nus_wide_test.h5') # 是一个 HDF5 格式的文件，用于存储 NUS-WIDE 数据集中测试集图像的特征和标签信息
        self.train_features = h5py.File(train_loc, 'r')
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, idx):
        file_name = self.image_names[idx]

        t = file_name.split("_")
        # path = os.path.join(self.src, "Flickr", "_".join(t[:-2]), t[-2] + "_" + t[-1])
        path = os.path.join(data_dir, "Flickr", "actor", "0001_2124494179.jpg")

        img = Image.open(path).convert('RGB')
        inputs = self.transforms(img)

        labels_1006 = np.int32(self.train_features.get(file_name + '-labels'))
        labels_81 = np.int32(self.train_features.get(file_name + '-labels_81'))

        # print("Loading labels for file:", file_name)
        # labels_81 = np.int32(self.train_features.get(file_name + '-labels_81'))
        # print("Loaded labels_81:", labels_81)

        return inputs, labels_1006, labels_81, file_name

    def __len__(self):
        return len(self.image_names)

# 指定要使用的GPU索引
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/data1/mazc/Xucy/CLIP/.cache/clip/ViT-B-32.pt", device)  #  process用于预处理image

# 设置数据预处理管道
transform = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  #  对于彩色图像，通常会分别计算每个通道的均值和标准差
])

# 设置数据集路径  nus_wide根目录路径
# data_dir = "/data1/mazc/Xucy/datasets/DATAROOT/NUS-WIDE/"
data_dir = "/data1/mazc/Xucy/datasets/DATAROOT/NUS-WIDE/"

# 加载概念标签列表
with open(os.path.join(data_dir, "Concepts81.txt"), "r") as file:
    concept_labels = [line.strip() for line in file.readlines()]

text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in concept_labels]).to(device)

# 创建命名空间对象，包含 data_path 属性
args = argparse.Namespace()
args.data_path = "/data1/mazc/Xucy/datasets/DATAROOT/NUS-WIDE/"

# 构建评估数据集
val_img_names = load_dict(os.path.join(args.data_path, 'test_img_names.pkl'))
val_dataset = ValDataset(args, val_img_names['img_names'], transform)

# start
# 从图像名列表中随机选择一小部分图像名，例如选择10个图像名
subset_size =1
num_images_in_val_dataset = len(val_dataset)
print("Number of images in validation dataset:", num_images_in_val_dataset)

subset_img_names = random.sample(val_img_names['img_names'], subset_size)
# 构建评估数据集的子集
val_dataset_subset = ValDataset(args, subset_img_names, transform)
# end

# 预测图像的标签
def predict_image_labels(image_path, similarity_threshold):
    image = Image.open(image_path).convert("RGB")
    processed_image = transform(image).unsqueeze(0).to(device)

    image_embedding = model.encode_image(processed_image).to(torch.float32)
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

    text_embedding = text_inputs.to(image_embedding.dtype)

    similarity_scores = (100.0 * image_embedding @ text_embedding.T).softmax(1)

    similarity_scores = similarity_scores[0]  # 获取第一个图像的相似性分数

    print("Similarity scores:", similarity_scores)  # 打印相似性分数

    # 仅当相似性分数高于阈值时才预测标签
    if similarity_scores.max() > similarity_threshold:
        predicted_label_indices = similarity_scores.argmax().item()
        predicted_labels = concept_labels[predicted_label_indices]

        print("Predicted label indices:", predicted_label_indices)  # 打印预测标签的索引
        print("Predicted label:", predicted_labels)  # 打印预测的标签
    else:
        predicted_labels = "Unknown"  # 或者你可以根据需求设置一个默认值

    return predicted_labels


# 对数据集中的图像进行预测并计算评估指标
true_labels = []           # 用于存储真实的标签
predicted_labels = []      # 用于存储预测的标签

for idx in range(len(val_dataset_subset)):
    inputs, labels_1006, labels_81, file_name = val_dataset_subset[idx]

    t = file_name.split("_")
    path = os.path.join(data_dir, "Flickr", "_".join(t[:-2]), t[-2] + "_" + t[-1])

    print("Constructed image path:", path)

    similarity_threshold = 0.3  # 设置相似性分数的阈值
    predicted_label = predict_image_labels(path, similarity_threshold)

    # 打印预测标签和真实标签
    print("Predicted label:", predicted_label)

    # 获取标签的索引
    label_indices = np.where(labels_81 == 1)[0]

    # 打印真实标签的索引和名称
    for label_index in label_indices:
        print(f"True label (labels_81) at index {label_index}: {concept_labels[label_index]}")
    # 确保预测标签在 concept_labels 中
    if predicted_label in concept_labels:
        true_labels.append(labels_81[idx])  # 使用labels_81作为真实标签
        predicted_labels.append(predicted_label)
    else:
        print(f"Predicted label '{predicted_label}' not found in concept_labels.")



# 将预测标签从字符串类型转换为数值类型
predicted_labels = [concept_labels.index(label) for label in predicted_labels]

# 计算评估指标
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='micro')

# 输出评估结果
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# 保存评估结果到文件
with open('/data1/mazc/cwy2/CLIP/CLIPx/CLIP/evaluation_results.txt', 'w') as f:
    f.write(f"准确率：{accuracy:.4f}\n")
    f.write(f"F1 分数：{f1:.4f}\n")
