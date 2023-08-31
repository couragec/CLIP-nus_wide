import os
import clip
import torch
from torchvision.datasets import CIFAR100
import numpy as np

np.set_printoptions(suppress=True)  # 设置numpy打印选项，禁用科学计数法
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/data1/mazc/Xucy/CLIP/.cache/clip/ViT-B-32.pt", device)

# Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("./data/"), download=True, train=False)
# print("test")
# # Specify the path to the CIFAR100 dataset
# data_path = "/data/cifar-100-python/"
#
# # Load the CIFAR100 dataset
# cifar100 = CIFAR100(root=data_path, download=False, train=False)

cifar100 = CIFAR100(root="/data1/mazc/cwy/data", download=True, train=False)
# cifar100 = CIFAR100(root=os.path.expanduser("./data/cifar-100-python/train"), download=False, train=False)
# print("test2")
# Prepare the inputs
# image, class_id = cifar100[3637]

image, class_id = cifar100[3000]
image_input = preprocess(image).unsqueeze(0).to(device)

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")