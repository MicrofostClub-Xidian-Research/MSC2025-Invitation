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

# ★★★ 新增：学习型编码参数 ★★★
# 学习型编码器的输出维度和内部结构
LEARNABLE_ENCODING_DIM = 64  # 学习型编码的输出维度
ENCODING_HIDDEN_DIM = 32     # 编码器的隐藏层维度
N_ENCODING_LAYERS = 2        # 编码器的隐藏层数量

print(f"Using device: {DEVICE}")


# ★★★ 新增：学习型编码器模块 ★★★
class LearnableEncoder(nn.Module):
    def __init__(self, input_dims, encoding_dim, hidden_dim, n_layers):
        super().__init__()
        self.input_dims = input_dims
        self.encoding_dim = encoding_dim
        self.output_dims = encoding_dim
        
        # 构建可学习的编码网络
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dims, hidden_dim))
        layers.append(nn.ReLU())
        
        # 隐藏层
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, encoding_dim))
        
        self.encoder_net = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Xavier初始化来保证训练稳定性"""
        for module in self.encoder_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x shape: [N, input_dims]
        # 通过可学习的网络进行编码
        encoded = self.encoder_net(x)
        return encoded


# ★★★ 新增：学习型三角编码器（另一种变体） ★★★
class LearnableTrigonometricEncoder(nn.Module):
    def __init__(self, input_dims, n_frequencies):
        super().__init__()
        self.input_dims = input_dims
        self.n_frequencies = n_frequencies
        
        # 可学习的频率参数
        self.frequencies = nn.Parameter(torch.randn(input_dims, n_frequencies))
        # 可学习的相位偏移
        self.phases = nn.Parameter(torch.rand(n_frequencies) * 2 * np.pi)
        
        # 输出维度: 每个频率产生sin和cos两个特征
        self.output_dims = input_dims * n_frequencies * 2
        
    def forward(self, x):
        # x shape: [N, input_dims]
        # frequencies shape: [input_dims, n_frequencies]
        
        # 计算 x @ frequencies + phases
        # x @ self.frequencies: [N, n_frequencies]
        projections = x @ self.frequencies + self.phases.unsqueeze(0)
        
        # 为每个输入维度计算sin和cos
        features = []
        for i in range(self.input_dims):
            # 对第i个输入维度的投影计算三角函数
            proj_i = x[:, i:i+1] @ self.frequencies[i:i+1, :] + self.phases.unsqueeze(0)
            sin_features = torch.sin(proj_i)
            cos_features = torch.cos(proj_i)
            features.extend([sin_features, cos_features])
        
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

# ★★★ 修改点 1: 选择学习型编码器类型 ★★★
# 选项1: 通用学习型编码器
encoder = LearnableEncoder(
    input_dims=2,
    encoding_dim=LEARNABLE_ENCODING_DIM,
    hidden_dim=ENCODING_HIDDEN_DIM,
    n_layers=N_ENCODING_LAYERS
).to(DEVICE)

# 选项2: 学习型三角编码器（注释掉选项1使用选项2）
# encoder = LearnableTrigonometricEncoder(
#     input_dims=2,
#     n_frequencies=16
# ).to(DEVICE)

coords_encoded = encoder(coords)  # 通过学习型编码器编码坐标

print(f"\nOriginal coords shape: {coords.shape}")
print(f"Learnable encoded coords shape: {coords_encoded.shape}")
print(f"Target pixels shape: {pixels.shape}")
print(f"Encoding output dimension: {encoder.output_dims}")


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
    in_features=encoder.output_dims,  # <-- 使用学习型编码后的维度
    hidden_features=HIDDEN_DIM,
    hidden_layers=N_HIDDEN_LAYERS,
    out_features=3
).to(DEVICE)

print("\nModel Architecture:")
print(f"Encoder: {encoder}")
print(f"Main Model: {model}")

# --- 5. 定义损失函数和优化器 ---
loss_fn = nn.MSELoss()

# ★★★ 关键点: 优化器需要同时优化编码器和主模型的参数 ★★★
all_parameters = list(encoder.parameters()) + list(model.parameters())
optimizer = torch.optim.Adam(all_parameters, lr=LEARNING_RATE)

print(f"\nTotal trainable parameters:")
encoder_params = sum(p.numel() for p in encoder.parameters())
model_params = sum(p.numel() for p in model.parameters())
print(f"Encoder parameters: {encoder_params:,}")
print(f"Main model parameters: {model_params:,}")
print(f"Total parameters: {encoder_params + model_params:,}")

# --- 6. 训练模型 ---
print("\nStarting training...")
start_time = time.time()

# 记录编码器参数的变化（用于可视化）
initial_encoder_state = {name: param.clone().detach() 
                        for name, param in encoder.named_parameters()}

for epoch in range(EPOCHS):
    # ★★★ 修改点 3: 在每个epoch中重新计算编码特征 ★★★
    # 这是关键！因为编码器参数在训练中会变化
    coords_encoded = encoder(coords)
    predicted_pixels = model(coords_encoded)
    
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
encoder.eval()
with torch.no_grad():
    # 重建时使用训练后的编码器
    coords_encoded_final = encoder(coords)
    reconstructed_pixels_standardized = model(coords_encoded_final)

# ★★★ 将标准化的输出逆向转换回 [0, 1] 范围以便显示 ★★★
reconstructed_pixels = reconstructed_pixels_standardized * std.view(1, -1) + mean.view(1, -1)
reconstructed_pixels.clamp_(0.0, 1.0)

reconstructed_img_tensor = reconstructed_pixels.view(H, W, 3)
reconstructed_img_np = reconstructed_img_tensor.cpu().numpy()
original_img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

# --- 8. 综合可视化 ---
plt.figure(figsize=(20, 10))

# 第一行：重建结果
plt.subplot(2, 4, 1)
plt.title("Original Image")
plt.imshow(original_img_np)
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title(f"Reconstructed Image\n(Learnable Encoding, Dim={LEARNABLE_ENCODING_DIM})")
plt.imshow(reconstructed_img_np)
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("Difference")
plt.imshow(np.abs(original_img_np - reconstructed_img_np))
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("Encoding Feature Distribution")
coords_encoded_np = coords_encoded_final.cpu().numpy()
plt.hist(coords_encoded_np.flatten(), bins=50, alpha=0.7)
plt.xlabel("Feature Value")
plt.ylabel("Frequency")

# 第二行：编码器参数分析
if isinstance(encoder, LearnableEncoder):
    # 分析全连接编码器的权重
    plt.subplot(2, 4, 5)
    first_layer_weights = encoder.encoder_net[0].weight.data.cpu().numpy()
    plt.imshow(first_layer_weights, cmap='RdBu', aspect='auto')
    plt.title("First Layer Weights")
    plt.colorbar()
    
    plt.subplot(2, 4, 6)
    last_layer_weights = encoder.encoder_net[-1].weight.data.cpu().numpy()
    plt.imshow(last_layer_weights.T, cmap='RdBu', aspect='auto')
    plt.title("Last Layer Weights")
    plt.colorbar()
    
elif isinstance(encoder, LearnableTrigonometricEncoder):
    # 分析三角编码器的频率参数
    plt.subplot(2, 4, 5)
    frequencies = encoder.frequencies.data.cpu().numpy()
    plt.scatter(frequencies[0, :], frequencies[1, :], alpha=0.7)
    plt.xlabel("X Frequencies")
    plt.ylabel("Y Frequencies")
    plt.title("Learned Frequencies")
    plt.grid(True)
    
    plt.subplot(2, 4, 6)
    phases = encoder.phases.data.cpu().numpy()
    plt.hist(phases, bins=20, alpha=0.7)
    plt.xlabel("Phase (radians)")
    plt.ylabel("Count")
    plt.title("Learned Phases")

# 参数变化分析
plt.subplot(2, 4, 7)
# 计算参数变化幅度
param_changes = []
for name, initial_param in initial_encoder_state.items():
    current_param = dict(encoder.named_parameters())[name]
    change = torch.norm(current_param - initial_param).item()
    param_changes.append(change)

plt.bar(range(len(param_changes)), param_changes)
plt.title("Parameter Change Magnitude")
plt.xlabel("Layer Index")
plt.ylabel("L2 Norm of Change")

plt.subplot(2, 4, 8)
# 编码特征的空间分布
sample_coords = coords[:1000:10]  # 采样一些坐标点
with torch.no_grad():  # 确保在无梯度模式下计算
    sample_encoded = encoder(sample_coords).detach().cpu().numpy()
plt.scatter(sample_coords[:, 0].detach().cpu().numpy(), sample_coords[:, 1].detach().cpu().numpy(), 
           c=sample_encoded[:, 0], cmap='viridis', alpha=0.7)
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("First Encoding Feature\nSpatial Distribution")
plt.colorbar()

plt.tight_layout()
plt.show()

# --- 9. 输出统计信息 ---
print(f"\nLearnable encoding statistics:")
print(f"Encoding feature mean: {coords_encoded_final.mean().item():.6f}")
print(f"Encoding feature std: {coords_encoded_final.std().item():.6f}")
print(f"Encoding feature range: [{coords_encoded_final.min().item():.6f}, {coords_encoded_final.max().item():.6f}]")

# 计算总的参数变化
total_change = sum(torch.norm(dict(encoder.named_parameters())[name] - initial_param).item() 
                  for name, initial_param in initial_encoder_state.items())
print(f"Total encoder parameter change: {total_change:.6f}")
