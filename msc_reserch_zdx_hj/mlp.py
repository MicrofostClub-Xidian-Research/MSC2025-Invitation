import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- 1. 超参数设置 ---
IMAGE_PATH = 'ye.png'
IMG_SIZE = 128
HIDDEN_DIM = 256
N_HIDDEN_LAYERS = 4
EPOCHS = 8000
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# 加载和预处理图像
img = Image.open(IMAGE_PATH).convert('RGB')
print(f"Successfully loaded image from: {IMAGE_PATH}")

img = TF.resize(img, (IMG_SIZE, IMG_SIZE))
img_tensor = TF.to_tensor(img).to(DEVICE) # shape: [3, H, W], 范围 [0, 1]

# ★★★ 2. 对像素值进行标准化 ★★★
# 计算每个通道的均值和标准差
# img_tensor shape is [C, H, W], so we calculate mean/std over H and W dimensions
mean = torch.mean(img_tensor, dim=[1, 2])
std = torch.std(img_tensor, dim=[1, 2])

# 为防止标准差为0（例如纯色通道）导致除零错误，增加一个极小值
std = torch.max(std, torch.tensor(1e-6).to(DEVICE)) 

print(f"\nImage stats (per channel):")
print(f"Mean: {mean.cpu().numpy()}")
print(f"Std:  {std.cpu().numpy()}")

# 应用标准化: (x - mean) / std
# 我们需要调整 mean 和 std 的形状以利用广播机制
img_tensor_standardized = (img_tensor - mean[:, None, None]) / std[:, None, None]

H, W = IMG_SIZE, IMG_SIZE
pixels = img_tensor_standardized.permute(1, 2, 0).view(-1, 3) # 使用标准化后的像素作为目标

# --- 3. 创建输入数据（坐标网格） ---
grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing='ij')
coords = torch.stack([
    grid_x / (W - 1) * 2 - 1,
    grid_y / (H - 1) * 2 - 1
], dim=-1).to(DEVICE)
coords = coords.view(-1, 2)

print(f"\nImage Size: {H}x{W}")
print(f"Input coordinates shape: {coords.shape}")
print(f"Target pixels shape: {pixels.shape}")


# --- 4. 构建 MLP 模型 ---
class MLPImageReconstructorLinear(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLPImageReconstructorLinear(
    in_features=2,
    hidden_features=HIDDEN_DIM,
    hidden_layers=N_HIDDEN_LAYERS,
    out_features=3
).to(DEVICE)

print("\nModel Architecture:")
print(model)

# --- 5. 定义损失函数和优化器 ---
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. 训练模型 ---
print("\nStarting training...")
start_time = time.time()
for epoch in range(EPOCHS):
    predicted_pixels = model(coords)
    loss = loss_fn(predicted_pixels, pixels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- 7. 重建与可视化 ---
model.eval()
with torch.no_grad():
    reconstructed_pixels_standardized = model(coords)

# ★★★ 将标准化的输出逆向转换回 [0, 1] 范围以便显示 ★★★
# 逆向操作: x_norm = x_std * std + mean
# 调整 mean 和 std 的形状以匹配 [N, C] 的像素列表
reconstructed_pixels = reconstructed_pixels_standardized * std.view(1, -1) + mean.view(1, -1)

# 重要：逆向转换后，数值可能略微超出[0,1]范围，需要裁剪
reconstructed_pixels.clamp_(0.0, 1.0)

reconstructed_img_tensor = reconstructed_pixels.view(H, W, 3)
reconstructed_img_np = reconstructed_img_tensor.cpu().numpy()
original_img_np = img_tensor.permute(1, 2, 0).cpu().numpy() # 原始图像依然使用[0,1]范围的张量

# 显示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img_np)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title(f"Reconstructed Image (After {EPOCHS} Epochs)")
plt.imshow(reconstructed_img_np)
plt.axis('off')
plt.show()