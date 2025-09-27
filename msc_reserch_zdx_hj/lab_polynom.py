import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- 1. 参数设置 ---
IMAGE_PATH = 'ye.png'
IMG_SIZE = 128
HIDDEN_DIM = 256
N_HIDDEN_LAYERS = 4
EPOCHS = 8000
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ★★★ 新增：多项式特征参数 ★★★
# 多项式的最高次数
POLYNOMIAL_DEGREE = 5

print(f"Using device: {DEVICE}")


# ★★★ 新增：多项式特征编码器模块 ★★★
class PolynomialEncoder(nn.Module):
    def __init__(self, input_dims, degree):
        super().__init__()
        self.input_dims = input_dims
        self.degree = degree
        
        # 计算多项式特征的输出维度
        # 对于二维输入 (x, y) 和度数 d，特征包括：
        # 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, ..., x^d, ..., y^d
        # 总数为 (d+1)(d+2)/2
        self.output_dims = (degree + 1) * (degree + 2) // 2
        
    def forward(self, x):
        # x shape: [N, 2] (假设输入是2D坐标)
        batch_size = x.shape[0]
        features = []
        
        # 生成所有次数组合的多项式特征
        for total_degree in range(self.degree + 1):
            for x_degree in range(total_degree + 1):
                y_degree = total_degree - x_degree
                # x[:, 0]^x_degree * x[:, 1]^y_degree
                feature = (x[:, 0] ** x_degree) * (x[:, 1] ** y_degree)
                features.append(feature.unsqueeze(1))
        
        # 拼接所有特征
        return torch.cat(features, dim=1)


# --- 2. 加载和预处理图像 ---
img = Image.open(IMAGE_PATH).convert('RGB')
print(f"Successfully loaded image from: {IMAGE_PATH}")

img = TF.resize(img, (IMG_SIZE, IMG_SIZE))
img_tensor = TF.to_tensor(img).to(DEVICE)

# ★★★ 标准化流程 ★★★
mean = torch.mean(img_tensor, dim=[1, 2])
std = torch.std(img_tensor, dim=[1, 2])
std = torch.max(std, torch.tensor(1e-6).to(DEVICE))

print(f"\nImage stats (per channel):")
print(f"Mean: {mean.cpu().numpy()}")
print(f"Std:  {std.cpu().numpy()}")

img_tensor_standardized = (img_tensor - mean[:, None, None]) / std[:, None, None]
H, W = IMG_SIZE, IMG_SIZE
pixels = img_tensor_standardized.permute(1, 2, 0).view(-1, 3)

# --- 3. 创建输入数据（坐标网格） ---
grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing='ij')
coords = torch.stack([
    grid_x / (W - 1) * 2 - 1,
    grid_y / (H - 1) * 2 - 1
], dim=-1).to(DEVICE)
coords = coords.view(-1, 2)

# ★★★ 修改点 1: 对坐标进行多项式特征编码 ★★★
encoder = PolynomialEncoder(input_dims=2, degree=POLYNOMIAL_DEGREE)
coords_encoded = encoder(coords)  # 新的、高维的坐标输入

print(f"\nOriginal coords shape: {coords.shape}")
print(f"Polynomial encoded coords shape: {coords_encoded.shape}")  # 维度显著增加
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

# ★★★ 修改点 2: 使用编码后的维度作为模型输入 ★★★
model = MLPImageReconstructorLinear(
    in_features=encoder.output_dims,  # <-- 使用多项式编码后的维度
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
    # ★★★ 修改点 3: 使用多项式编码后的坐标进行训练 ★★★
    predicted_pixels = model(coords_encoded)  # <-- 使用编码后的坐标
    
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
    # 重建时同样需要对坐标进行多项式编码
    reconstructed_pixels_standardized = model(coords_encoded)

# ★★★ 将标准化的输出逆向转换回 [0, 1] 范围以便显示 ★★★
reconstructed_pixels = reconstructed_pixels_standardized * std.view(1, -1) + mean.view(1, -1)
reconstructed_pixels.clamp_(0.0, 1.0)

reconstructed_img_tensor = reconstructed_pixels.view(H, W, 3)
reconstructed_img_np = reconstructed_img_tensor.cpu().numpy()
original_img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

# 显示
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_img_np)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title(f"Reconstructed Image\n(Polynomial Features, Degree={POLYNOMIAL_DEGREE})")
plt.imshow(reconstructed_img_np)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(np.abs(original_img_np - reconstructed_img_np))
plt.axis('off')
plt.tight_layout()
plt.show()