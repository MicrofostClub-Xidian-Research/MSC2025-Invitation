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

# ★★★ 新增：高斯核特征参数 ★★★
# 高斯核的数量和带宽参数
N_GAUSSIAN_KERNELS = 64  # 高斯核的数量
GAUSSIAN_BANDWIDTH = 0.1  # 高斯核的带宽(标准差)

print(f"Using device: {DEVICE}")


# ★★★ 新增：高斯核特征编码器模块 ★★★
class GaussianKernelEncoder(nn.Module):
    def __init__(self, input_dims, n_kernels, bandwidth):
        super().__init__()
        self.input_dims = input_dims
        self.n_kernels = n_kernels
        self.bandwidth = bandwidth
        
        # 随机初始化高斯核的中心点
        # 在输入空间[-1, 1]^2中均匀分布
        self.register_buffer(
            'centers', 
            torch.rand(n_kernels, input_dims) * 2 - 1  # 范围 [-1, 1]
        )
        
        # 输出维度等于高斯核的数量
        self.output_dims = n_kernels
        
    def forward(self, x):
        # x shape: [N, input_dims]
        # centers shape: [n_kernels, input_dims]
        
        # 计算每个输入点到每个高斯核中心的距离
        # x.unsqueeze(1): [N, 1, input_dims]
        # self.centers.unsqueeze(0): [1, n_kernels, input_dims]
        # distances: [N, n_kernels, input_dims]
        distances = x.unsqueeze(1) - self.centers.unsqueeze(0)
        
        # 计算欧几里得距离的平方
        # squared_distances: [N, n_kernels]
        squared_distances = torch.sum(distances ** 2, dim=2)
        
        # 应用高斯核函数: exp(-d^2 / (2 * sigma^2))
        gaussian_features = torch.exp(-squared_distances / (2 * self.bandwidth ** 2))
        
        return gaussian_features


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

# ★★★ 修改点 1: 对坐标进行高斯核特征编码 ★★★
encoder = GaussianKernelEncoder(
    input_dims=2, 
    n_kernels=N_GAUSSIAN_KERNELS, 
    bandwidth=GAUSSIAN_BANDWIDTH
)
encoder = encoder.to(DEVICE)
coords_encoded = encoder(coords)  # 新的、高维的坐标输入

print(f"\nOriginal coords shape: {coords.shape}")
print(f"Gaussian kernel encoded coords shape: {coords_encoded.shape}")  # 维度显著增加
print(f"Target pixels shape: {pixels.shape}")
print(f"Number of Gaussian kernels: {N_GAUSSIAN_KERNELS}")
print(f"Gaussian bandwidth: {GAUSSIAN_BANDWIDTH}")


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
    in_features=encoder.output_dims,  # <-- 使用高斯核编码后的维度
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
    # ★★★ 修改点 3: 使用高斯核编码后的坐标进行训练 ★★★
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
    # 重建时同样需要对坐标进行高斯核编码
    reconstructed_pixels_standardized = model(coords_encoded)

# ★★★ 将标准化的输出逆向转换回 [0, 1] 范围以便显示 ★★★
reconstructed_pixels = reconstructed_pixels_standardized * std.view(1, -1) + mean.view(1, -1)
reconstructed_pixels.clamp_(0.0, 1.0)

reconstructed_img_tensor = reconstructed_pixels.view(H, W, 3)
reconstructed_img_np = reconstructed_img_tensor.cpu().numpy()
original_img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

# 显示结果和高斯核中心可视化
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(original_img_np)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title(f"Reconstructed Image\n(Gaussian Kernels, N={N_GAUSSIAN_KERNELS})")
plt.imshow(reconstructed_img_np)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Difference")
plt.imshow(np.abs(original_img_np - reconstructed_img_np))
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Gaussian Kernel Centers")
centers_np = encoder.centers.cpu().numpy()
plt.scatter(centers_np[:, 0], centers_np[:, 1], c='red', alpha=0.6, s=20)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 8. 可选：可视化高斯核特征的激活模式 ---
print(f"\nGaussian kernel feature statistics:")
print(f"Feature mean: {coords_encoded.mean().item():.4f}")
print(f"Feature std: {coords_encoded.std().item():.4f}")
print(f"Feature max: {coords_encoded.max().item():.4f}")
print(f"Feature min: {coords_encoded.min().item():.4f}")
