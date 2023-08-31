"""
@Project ：CLIP 
@File    ：nuswide_zl_evl.py
@IDE     ：PyCharm 
@Author  ：Xucy
@Date    ：2023/8/15 12:43 
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip
from sklearn.metrics import accuracy_score, f1_score

# 加载CLIP模型和令牌
from utils.dataset import ValDataset, load_dict
from utils.transforms import build_transform


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/data1/mazc/Xucy/CLIP/.cache/clip/ViT-B-32.pt", device)

# 设置数据预处理管道
transform = Compose([
    Resize((224, 224), interpolation=Image.BICUBIC),
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
text_input = ["a photo of " + label for label in concept_labels]
text_input = torch.cat([model.encode_text(text).unsqueeze(0) for text in text_input]).to(device)

# 设置数据预处理管道
transform = build_transform(False, 224)  # 注意这里的args应该设置为适当的值

# 构建评估数据集
val_img_names = load_dict(os.path.join(data_dir, 'test_img_names.pkl'))
val_dataset = ValDataset(data_dir, val_img_names['img_names'], transform)

# 预测图像的标签
def predict_image_labels(image_path):
    image = Image.open(image_path).convert("RGB")
    processed_image = transform(image).unsqueeze(0).to(device)
    image_embedding = model.encode_image(processed_image)

    # 计算相似性分数
    similarity_scores = (100.0 * image_embedding @ text_input.T).softmax(1)

    # 预测标签
    predicted_label_indices = np.argsort(similarity_scores.cpu().numpy(), axis=1)[:, ::-1][:, :5]
    predicted_labels = [concept_labels[i] for i in predicted_label_indices]

    return predicted_labels

# 对数据集中的图像进行预测并计算评估指标
true_labels = []
predicted_labels = []

for idx in range(len(val_dataset)):
    inputs, labels_1006, labels_81, file_name = val_dataset[idx]
    predicted_label = predict_image_labels(os.path.join(data_dir, 'Flickr', file_name))[0]  # 获取预测的第一个标签
    true_labels.append(labels_81[0])  # 使用labels_81作为真实标签
    predicted_labels.append(predicted_label)

# 计算评估指标
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='micro')

# 输出评估结果
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
