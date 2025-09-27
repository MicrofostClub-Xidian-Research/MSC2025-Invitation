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

# ★★★ 新增：随机傅里叶特征参数 ★★★
# RFF的特征数量和频率标准差
N_RFF_FEATURES = 256  # 增加特征数量
RFF_GAMMA = 10.0      # 增加频率参数，捕获更多高频细节

print(f"Using device: {DEVICE}")


# ★★★ 新增：随机傅里叶特征编码器模块 ★★★
class RandomFourierFeatureEncoder(nn.Module):
    def __init__(self, input_dims, n_features, gamma):
        super().__init__()
        self.input_dims = input_dims
        self.n_features = n_features
        self.gamma = gamma
        
        # 确保特征数量为偶数（sin和cos成对出现）
        if n_features % 2 != 0:
            n_features += 1
            self.n_features = n_features
        
        # 改进的频率采样策略
        # 使用多尺度频率：低频 + 中频 + 高频
        n_half = n_features // 2
        n_low = n_half // 3      # 低频
        n_mid = n_half // 3      # 中频  
        n_high = n_half - n_low - n_mid  # 高频
        
        # 多尺度频率权重
        W_low = torch.randn(input_dims, n_low) * (gamma * 0.1)    # 低频
        W_mid = torch.randn(input_dims, n_mid) * (gamma * 0.5)    # 中频
        W_high = torch.randn(input_dims, n_high) * gamma          # 高频
        
        W = torch.cat([W_low, W_mid, W_high], dim=1)
        self.register_buffer('W', W)
        
        # 固定相位偏移为0，简化学习
        self.register_buffer(
            'b',
            torch.zeros(n_features // 2)
        )
        
        # 输出维度
        self.output_dims = n_features
        
    def forward(self, x):
        # x shape: [N, input_dims]
        # W shape: [input_dims, n_features//2]
        # x @ W: [N, n_features//2]
        
        # 计算 x * W + b
        projections = x @ self.W + self.b.unsqueeze(0)
        
        # 改进的特征计算
        # 使用更平衡的缩放因子
        scale = np.sqrt(1.0 / self.n_features)
        
        cos_features = torch.cos(projections) * scale
        sin_features = torch.sin(projections) * scale
        
        # 拼接cos和sin特征
        features = torch.cat([cos_features, sin_features], dim=1)
        
        # 可选：添加特征归一化以稳定训练
        # features = features / (features.std() + 1e-8)
        
        return features


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

# ★★★ 修改点 1: 对坐标进行随机傅里叶特征编码 ★★★
encoder = RandomFourierFeatureEncoder(
    input_dims=2, 
    n_features=N_RFF_FEATURES, 
    gamma=RFF_GAMMA
)
encoder = encoder.to(DEVICE)
coords_encoded = encoder(coords)  # 新的、高维的坐标输入

print(f"\nOriginal coords shape: {coords.shape}")
print(f"Random Fourier Feature encoded coords shape: {coords_encoded.shape}")
print(f"Target pixels shape: {pixels.shape}")
print(f"Number of RFF features: {N_RFF_FEATURES}")
print(f"RFF gamma parameter: {RFF_GAMMA}")


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
    in_features=encoder.output_dims,  # <-- 使用RFF编码后的维度
    hidden_features=HIDDEN_DIM,
    hidden_layers=N_HIDDEN_LAYERS,
    out_features=3
).to(DEVICE)

print("\nModel Architecture:")
print(model)

# --- 5. 定义损失函数和优化器 ---
loss_fn = nn.MSELoss()
# 对于RFF，可能需要更小的学习率
RFF_LEARNING_RATE = LEARNING_RATE * 0.5  # 降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=RFF_LEARNING_RATE)

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=500
)

# --- 6. 训练模型 ---
print("\nStarting training...")
start_time = time.time()
for epoch in range(EPOCHS):
    # ★★★ 修改点 3: 使用RFF编码后的坐标进行训练 ★★★
    predicted_pixels = model(coords_encoded)  # <-- 使用编码后的坐标
    
    loss = loss_fn(predicted_pixels, pixels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 更新学习率
    if (epoch + 1) % 100 == 0:
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 手动检测学习率变化
        if current_lr != prev_lr:
            print(f"Learning rate reduced from {prev_lr:.2e} to {current_lr:.2e}")
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, LR: {current_lr:.2e}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- 7. 重建与可视化 ---
model.eval()
with torch.no_grad():
    # 重建时同样需要对坐标进行RFF编码
    reconstructed_pixels_standardized = model(coords_encoded)

# ★★★ 将标准化的输出逆向转换回 [0, 1] 范围以便显示 ★★★
reconstructed_pixels = reconstructed_pixels_standardized * std.view(1, -1) + mean.view(1, -1)
reconstructed_pixels.clamp_(0.0, 1.0)

reconstructed_img_tensor = reconstructed_pixels.view(H, W, 3)
reconstructed_img_np = reconstructed_img_tensor.cpu().numpy()
original_img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

# 显示结果和RFF参数可视化
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(original_img_np)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title(f"Reconstructed Image\n(RFF, N={N_RFF_FEATURES}, γ={RFF_GAMMA})")
plt.imshow(reconstructed_img_np)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Difference")
plt.imshow(np.abs(original_img_np - reconstructed_img_np))
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("RFF Weight Distribution")
W_np = encoder.W.cpu().numpy()
plt.scatter(W_np[0, :], W_np[1, :], alpha=0.6, s=20, c='blue')
plt.xlabel('W_x (X-direction weights)')
plt.ylabel('W_y (Y-direction weights)')
plt.title(f'Random Weights Distribution\n(γ={RFF_GAMMA})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 8. 可选：分析RFF特征的性质 ---
print(f"\nRandom Fourier Feature statistics:")
print(f"Feature mean: {coords_encoded.mean().item():.6f}")
print(f"Feature std: {coords_encoded.std().item():.6f}")
print(f"Feature max: {coords_encoded.max().item():.6f}")
print(f"Feature min: {coords_encoded.min().item():.6f}")

# 可视化RFF特征的频谱特性
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(coords_encoded.cpu().numpy().flatten(), bins=50, alpha=0.7)
plt.title("RFF Feature Value Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
W_magnitudes = torch.norm(encoder.W, dim=0).cpu().numpy()
plt.hist(W_magnitudes, bins=30, alpha=0.7, color='orange')
plt.title("RFF Weight Magnitude Distribution")
plt.xlabel("||W||")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
phase_offsets = encoder.b.cpu().numpy()
plt.hist(phase_offsets, bins=30, alpha=0.7, color='green')
plt.title("Phase Offset Distribution")
plt.xlabel("Phase (radians)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
