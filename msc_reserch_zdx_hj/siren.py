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

# ★★★ 新增：SIREN特有参数 ★★★
# ω0: 控制正弦激活函数的频率
OMEGA_0 = 30.0  # 第一层的频率参数
OMEGA_HIDDEN = 1.0  # 隐藏层的频率参数

print(f"Using device: {DEVICE}")


# ★★★ 新增：正弦激活函数 ★★★
class SinActivation(nn.Module):
    def __init__(self, omega=1.0):
        super().__init__()
        self.omega = omega
        
    def forward(self, x):
        return torch.sin(self.omega * x)


# ★★★ 新增：SIREN模型架构 ★★★
class SirenMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, omega_0=30.0, omega_hidden=1.0):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_0 = omega_0
        self.omega_hidden = omega_hidden
        
        # 第一层（特殊初始化）
        self.first_layer = nn.Linear(in_features, hidden_features)
        self.first_activation = SinActivation(omega_0)
        
        # 隐藏层
        self.hidden_layers_list = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()
        
        for _ in range(hidden_layers):
            layer = nn.Linear(hidden_features, hidden_features)
            activation = SinActivation(omega_hidden)
            self.hidden_layers_list.append(layer)
            self.hidden_activations.append(activation)
        
        # 输出层（无激活函数）
        self.output_layer = nn.Linear(hidden_features, out_features)
        
        # SIREN特有的权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """SIREN论文中的特殊初始化方法"""
        # 第一层：从Uniform(-1/in_features, 1/in_features)中采样
        with torch.no_grad():
            self.first_layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
        
        # 隐藏层：从Uniform(-sqrt(6/hidden_features)/omega, sqrt(6/hidden_features)/omega)中采样
        for layer in self.hidden_layers_list:
            with torch.no_grad():
                bound = np.sqrt(6 / self.hidden_features) / self.omega_hidden
                layer.weight.uniform_(-bound, bound)
        
        # 输出层：标准初始化
        with torch.no_grad():
            bound = np.sqrt(6 / self.hidden_features)
            self.output_layer.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        # 第一层
        x = self.first_layer(x)
        x = self.first_activation(x)
        
        # 隐藏层
        for layer, activation in zip(self.hidden_layers_list, self.hidden_activations):
            x = layer(x)
            x = activation(x)
        
        # 输出层
        x = self.output_layer(x)
        return x


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

print(f"\nImage Size: {H}x{W}")
print(f"Input coordinates shape: {coords.shape}")
print(f"Target pixels shape: {pixels.shape}")

# ★★★ SIREN使用原始坐标，不需要额外的位置编码 ★★★
print(f"SIREN uses raw coordinates without positional encoding")
print(f"First layer frequency (ω₀): {OMEGA_0}")
print(f"Hidden layer frequency (ω): {OMEGA_HIDDEN}")

# --- 4. 构建 SIREN 模型 ---
model = SirenMLP(
    in_features=2,  # 直接使用2D坐标
    hidden_features=HIDDEN_DIM,
    hidden_layers=N_HIDDEN_LAYERS,
    out_features=3,  # RGB三个通道
    omega_0=OMEGA_0,
    omega_hidden=OMEGA_HIDDEN
).to(DEVICE)

print("\nSIREN Model Architecture:")
print(model)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")

# --- 5. 定义损失函数和优化器 ---
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. 训练模型 ---
print("\nStarting SIREN training...")
start_time = time.time()

# 记录训练损失
train_losses = []

for epoch in range(EPOCHS):
    # SIREN直接使用原始坐标
    predicted_pixels = model(coords)
    loss = loss_fn(predicted_pixels, pixels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- 7. 重建与可视化 ---
model.eval()
with torch.no_grad():
    reconstructed_pixels_standardized = model(coords)

# ★★★ 将标准化的输出逆向转换回 [0, 1] 范围以便显示 ★★★
reconstructed_pixels = reconstructed_pixels_standardized * std.view(1, -1) + mean.view(1, -1)
reconstructed_pixels.clamp_(0.0, 1.0)

reconstructed_img_tensor = reconstructed_pixels.view(H, W, 3)
reconstructed_img_np = reconstructed_img_tensor.cpu().numpy()
original_img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

# --- 8. 综合可视化 ---
plt.figure(figsize=(20, 10))

# 第一行：重建结果对比
plt.subplot(2, 4, 1)
plt.title("Original Image")
plt.imshow(original_img_np)
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title(f"SIREN Reconstructed\n(ω₀={OMEGA_0}, ω={OMEGA_HIDDEN})")
plt.imshow(reconstructed_img_np)
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("Difference")
plt.imshow(np.abs(original_img_np - reconstructed_img_np))
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("Training Loss")
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 第二行：SIREN特有的分析
plt.subplot(2, 4, 5)
# 可视化第一层权重
first_layer_weights = model.first_layer.weight.data.cpu().numpy()
plt.imshow(first_layer_weights, cmap='RdBu', aspect='auto')
plt.title(f"First Layer Weights\n(ω₀={OMEGA_0})")
plt.colorbar()

plt.subplot(2, 4, 6)
# 可视化隐藏层权重（选择第一个隐藏层）
if len(model.hidden_layers_list) > 0:
    hidden_weights = model.hidden_layers_list[0].weight.data.cpu().numpy()
    plt.imshow(hidden_weights, cmap='RdBu', aspect='auto')
    plt.title(f"Hidden Layer Weights\n(ω={OMEGA_HIDDEN})")
    plt.colorbar()

plt.subplot(2, 4, 7)
# 激活函数响应分析
x_range = torch.linspace(-2, 2, 1000)
sin_omega_0 = torch.sin(OMEGA_0 * x_range)
sin_omega_hidden = torch.sin(OMEGA_HIDDEN * x_range)

plt.plot(x_range.numpy(), sin_omega_0.numpy(), label=f'sin({OMEGA_0}x)', linewidth=2)
plt.plot(x_range.numpy(), sin_omega_hidden.numpy(), label=f'sin({OMEGA_HIDDEN}x)', linewidth=2)
plt.xlabel("Input")
plt.ylabel("Activation")
plt.title("SIREN Activation Functions")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 4, 8)
# 预测值的分布
plt.hist(reconstructed_pixels_standardized.detach().cpu().numpy().flatten(), 
         bins=50, alpha=0.7, label='SIREN Output')
plt.hist(pixels.detach().cpu().numpy().flatten(), 
         bins=50, alpha=0.7, label='Ground Truth')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Output Distribution")
plt.legend()

plt.tight_layout()
plt.show()

# --- 9. 性能统计 ---
print(f"\nSIREN Performance Statistics:")
print(f"Final training loss: {train_losses[-1]:.6f}")
print(f"Output value range: [{reconstructed_pixels_standardized.min().item():.6f}, {reconstructed_pixels_standardized.max().item():.6f}]")

# 计算重建质量指标
mse_error = torch.mean((reconstructed_pixels - img_tensor.permute(1, 2, 0).view(-1, 3)) ** 2).item()
print(f"Reconstruction MSE: {mse_error:.6f}")

# PSNR计算
psnr = 20 * np.log10(1.0 / np.sqrt(mse_error)) if mse_error > 0 else float('inf')
print(f"PSNR: {psnr:.2f} dB")

# --- 10. 与其他方法的理论对比 ---
print(f"\n=== SIREN vs Other Methods ===")
print(f"Method             | Input Dim | Activation  | Encoding")
print(f"Original MLP       |     2     |    ReLU     | None")
print(f"Positional MLP     |    40     |    ReLU     | Trigonometric")
print(f"SIREN (this)       |     2     |    Sin      | None (implicit)")
print(f"\nSIREN Advantages:")
print(f"- Natural bias towards smooth, differentiable functions")
print(f"- Excellent for representing natural images")
print(f"- No need for explicit positional encoding")
print(f"- Superior gradient properties for coordinate-based tasks")
