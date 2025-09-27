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

# ★★★ 新增：位置编码参数 ★★★
# L: 决定了编码后维度的参数。维度 = 2 * 输入维度 * L
# 越高的 L 能表示越高的频率
N_ENCODING_FUNCTIONS = 10 

print(f"Using device: {DEVICE}")


# ★★★ 新增：位置编码器模块 ★★★
class PositionalEncoder(nn.Module):
    def __init__(self, input_dims, num_functions):
        super().__init__()
        self.input_dims = input_dims
        self.num_functions = num_functions
        
        # 创建频率列表 [1, 2, 4, 8, ..., 2^(L-1)]
        self.freq_bands = 2.0 ** torch.arange(num_functions)
        
        # 计算编码后的输出维度
        # 对于每个输入维度(x, y)，每个频率都产生sin和cos两个值
        self.output_dims = input_dims * num_functions * 2

    def forward(self, x):
        # x shape: [N, input_dims]
        # unsqueeze(dim=-1) -> [N, input_dims, 1]
        # self.freq_bands -> [num_functions]
        # x * self.freq_bands -> [N, input_dims, num_functions]
        scaled_inputs = x.unsqueeze(-1) * self.freq_bands.to(x.device)
        
        # 将 sin 和 cos 的结果拼接起来
        # torch.sin(scaled_inputs) -> [N, input_dims, num_functions]
        # torch.cos(scaled_inputs) -> [N, input_dims, num_functions]
        # shape -> [N, input_dims, num_functions * 2]
        encoded = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1)
        
        # 展平为 [N, output_dims]
        return torch.flatten(encoded, start_dim=1)


# --- 2. 加载和预处理图像 ---
img = Image.open(IMAGE_PATH).convert('RGB')
print(f"Successfully loaded image from: {IMAGE_PATH}")

img = TF.resize(img, (IMG_SIZE, IMG_SIZE))
img_tensor = TF.to_tensor(img).to(DEVICE)

# 标准化流程保持不变
mean = torch.mean(img_tensor, dim=[1, 2])
std = torch.std(img_tensor, dim=[1, 2])
std = torch.max(std, torch.tensor(1e-6).to(DEVICE))
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

# ★★★ 修改点 1: 对坐标进行位置编码 ★★★
encoder = PositionalEncoder(input_dims=2, num_functions=N_ENCODING_FUNCTIONS)
coords_encoded = encoder(coords) # 新的、高维的坐标输入

print(f"\nOriginal coords shape: {coords.shape}")
print(f"Encoded coords shape: {coords_encoded.shape}") # 维度显著增加
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
    in_features=encoder.output_dims, # <-- 使用编码后的维度
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
    # ★★★ 修改点 3: 使用编码后的坐标进行训练 ★★★
    predicted_pixels = model(coords_encoded) # <-- 使用编码后的坐标
    
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
    # 重建时同样需要对坐标进行编码
    reconstructed_pixels_standardized = model(coords_encoded)

# 逆向标准化流程保持不变
reconstructed_pixels = reconstructed_pixels_standardized * std.view(1, -1) + mean.view(1, -1)
reconstructed_pixels.clamp_(0.0, 1.0)
reconstructed_img_tensor = reconstructed_pixels.view(H, W, 3)
reconstructed_img_np = reconstructed_img_tensor.cpu().numpy()
original_img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

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