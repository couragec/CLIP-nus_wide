"""
@Project ：CLIP 
@File    ：nuswide_zl_evl1.py
@IDE     ：PyCharm 
@Author  ：Xucy
@Date    ：2023/8/15 12:43 
"""
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


# 加载CLIP模型和令牌


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict


class ValDataset(Dataset):

    def __init__(self, args, image_names, transforms):
        self.src = args.data_path
        train_loc = os.path.join(self.src, 'features', 'nus_wide_test.h5')
        self.train_features = h5py.File(train_loc, 'r')
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, idx):
        file_name = self.image_names[idx]

        t = file_name.split("_")
        path = os.path.join(self.src, "Flickr", "_".join(t[:-2]), t[-2] + "_" + t[-1])

        img = Image.open(path).convert('RGB')
        inputs = self.transforms(img)

        labels_1006 = np.int32(self.train_features.get(file_name + '-labels'))
        labels_81 = np.int32(self.train_features.get(file_name + '-labels_81'))

        return inputs, labels_1006, labels_81, file_name

    def __len__(self):
        return len(self.image_names)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/data1/mazc/Xucy/CLIP/.cache/clip/ViT-B-32.pt", device)

# 设置数据预处理管道
# interpolation=PIL.Image.BICUBIC/interpolation=Image.BICUBIC
transform = Compose([
    Resize((224, 224), interpolation=PIL.Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# 设置数据集路径
data_dir = "/data1/mazc/Xucy/datasets/DATAROOT/NUS-WIDE/"

# 加载概念标签列表
with open(os.path.join(data_dir, "Concepts81.txt"), "r") as file:
    concept_labels = [line.strip() for line in file.readlines()]

# 预处理文本标签
# text_input = ["a photo of " + label for label in concept_labels]
# text_input = torch.cat([model.encode_text(text).unsqueeze(0) for text in text_input]).to(device)

text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in concept_labels]).to(device)

# 创建命名空间对象，包含 data_path 属性
args = argparse.Namespace()
args.data_path = "/data1/mazc/Xucy/datasets/DATAROOT/NUS-WIDE/"

# 构建评估数据集
val_img_names = load_dict(os.path.join(args.data_path, 'test_img_names.pkl'))
val_dataset = ValDataset(args, val_img_names['img_names'], transform)

# start
# 从图像名列表中随机选择一小部分图像名，例如选择10个图像名
subset_size = 10
subset_img_names = random.sample(val_img_names['img_names'], subset_size)
# 构建评估数据集的子集
val_dataset_subset = ValDataset(args, subset_img_names, transform)


# end

# 预测图像的标签
def predict_image_labels(image_path):
    image = Image.open(image_path).convert("RGB")
    processed_image = transform(image).unsqueeze(0).to(device)
    image_embedding = model.encode_image(processed_image)
    # image_embedding = model.encode_image(processed_image).to(torch.float32)
    text_embedding = model.encode_text(text_inputs)

    # 对图像嵌入进行标准化
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    # 对文本输入进行标准化
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    # 计算相似性分数
    # similarity_scores = (100.0 * image_embedding @ text_inputs.T).softmax(dim=-1)
    similarity_scores = (100.0 * image_embedding @ text_embedding.T).softmax(1)

    # 预测标签
    # predicted_label_indices = np.argsort(similarity_scores.cpu().numpy(), axis=1)[:, ::-1][:, :5]
    # predicted_labels = [concept_labels[i] for i in predicted_label_indices]
    predicted_label_indices = similarity_scores.argmax()
    predicted_labels = concept_labels[predicted_label_indices]

    #test
    print("Image path:", image_path)
    print("Image shape:", processed_image.shape)
    print("Text embedding shape:", text_embedding.shape)
    print("Image embedding shape:", image_embedding.shape)
    print("Similarity scores shape:", similarity_scores.shape)

    predicted_label_indices = similarity_scores.argmax()
    print("Predicted label indices:", predicted_label_indices)


    return predicted_labels


# # 对数据集中的图像进行预测并计算评估指标
# true_labels = []
# predicted_labels = []
#
# # for idx in range(len(val_dataset)):
# for idx in range(len(val_dataset_subset)):
#     # inputs, labels_1006, labels_81, file_name = val_dataset[idx]
#     # val_dataset_subset
#     inputs, labels_1006, labels_81, file_name = val_dataset_subset[idx]
#
#     t = file_name.split("_")
#     path = os.path.join(data_dir, "Flickr", "_".join(t[:-2]), t[-2] + "_" + t[-1])
#
#     print("Constructed image path:", path)
#
#     predicted_label = predict_image_labels(path)  # 获取预测的第一个标签
#     # predicted_label = predict_image_labels(os.path.join(data_dir, 'Flickr', file_name))[0]  # 获取预测的第一个标签
#
#     # true_labels.append(labels_81[0])  # 使用labels_81作为真实标签
#     # predicted_labels.append(predicted_label)
#
#     # 确保预测标签在concept_labels中
#     if predicted_label in concept_labels:
#         true_labels.append(labels_81[idx])  # 使用labels_81作为真实标签
#         predicted_labels.append(predicted_label)
#     else:
#         print(f"Predicted label '{predicted_label}' not found in concept_labels.")

# 将预测标签从字符串类型转换为数值类型
predicted_labels = [concept_labels.index(label) for label in predicted_labels]

# 计算评估指标
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='micro')

# 输出评估结果
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
# 保存评估结果到文件
with open('/data1/mazc/Xucy/CLIP/evaluation_results.txt', 'w') as f:
    f.write(f"准确率：{accuracy:.4f}\n")
    f.write(f"F1 分数：{f1:.4f}\n")
