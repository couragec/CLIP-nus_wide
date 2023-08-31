"""
@Project ：CLIP 
@File    ：nuswide_evl_zl.py
@IDE     ：PyCharm 
@Author  ：Xucy
@Date    ：2023/8/15 13:51 
"""
import torch
from PIL import Image
from torchvision import transforms
import pickle
from sklearn.metrics import accuracy_score, f1_score
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/data1/mazc/Xucy/CLIP/.cache/clip/ViT-B-32.pt", device)


# 设置文件路径
nuswide_dir = '/data1/mazc/Xucy/datasets/DATAROOT/NUS-WIDE/'
concepts_file = 'Concepts81.txt'
test_img_names_file = 'test_img_names.pkl'
test_labels_file = 'test_labels.pkl'

# 加载零样本标签
with open(nuswide_dir + concepts_file, 'r', encoding='utf-8') as f:
    zero_shot_labels = f.read().splitlines()

# 加载图像名称和真实标签
with open(nuswide_dir + test_img_names_file, 'rb') as f:
    test_img_names = pickle.load(f)
with open(nuswide_dir + test_labels_file, 'rb') as f:
    true_labels = pickle.load(f)


# 创建函数计算相似度分数
def compute_similarity(image, text):
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity_scores


# 进行零样本评估
all_predictions = []
for img_name, true_label in zip(test_img_names, true_labels):
    img_path = nuswide_dir + 'Flickr/' + img_name
    image = Image.open(img_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    similarity_scores = compute_similarity(image, zero_shot_labels)

    # 应用分类阈值或排序逻辑以获得预测结果
    predictions = ...  # 根据您的逻辑进行调整
    all_predictions.append(predictions)

# 计算评估指标
accuracy = accuracy_score(true_labels, all_predictions)
f1 = f1_score(true_labels, all_predictions, average='macro')

# 显示或保存评估结果
print(f"准确率：{accuracy:.4f}")
print(f"F1 分数：{f1:.4f}")
# 保存评估结果到文件
with open('evaluation_results.txt', 'w') as f:
    f.write(f"准确率：{accuracy:.4f}\n")
    f.write(f"F1 分数：{f1:.4f}\n")
