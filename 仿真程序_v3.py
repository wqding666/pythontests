import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# =================配置部分=================
class Config:
    def __init__(self):
        # 物理参数
        self.wavelength = 0.532e-6  # 532nm
        self.dx = 8.0e-6            # 像素尺寸 8um
        self.dy = 8.0e-6
        self.layer_distances = [0.05, 0.05, 0.05] # 层间距 (m)
        self.z_det = 0.05           # 最后一层到探测面的距离
        
        # 相干性参数
        self.l_train = 0.2e-3       # 训练时的相干长度 0.2mm
        self.n_modes = 9            # 模拟部分相干光所需的模式数量
        
        # 网络结构
        self.input_size = 28
        self.num_classes = 10
        
        # 训练参数
        self.batch_size = 64
        self.lr = 0.01
        self.epochs = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# =================物理引擎：角谱法传播=================
def angular_spectrum_propagator(u_in, z, wavelength, dx, dy, device):
    """
    基于角谱法的单色光传播 (PyTorch GPU 版本)
    u_in: [B, C, H, W] 复数场
    """
    B, C, H, W = u_in.shape
    
    # 1. 创建频率坐标 (必须在同一设备上)
    fx = torch.fft.fftfreq(H, d=dx).to(device)
    fy = torch.fft.fftfreq(W, d=dy).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    
    # 2. 传递函数 H(fx, fy)
    k = 2 * np.pi / wavelength
    arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    
    # 处理倏逝波
    mask = arg >= 0
    sqrt_arg = torch.sqrt(torch.complex(arg * mask, torch.zeros_like(arg)))
    
    H_transfer = torch.exp(1j * k * z * sqrt_arg)
    # 扩展维度以匹配 [B, C, H, W]: [1, 1, H, W]
    H_transfer = H_transfer.unsqueeze(0).unsqueeze(0) 
    
    # 3. 傅里叶变换 -> 频域乘积 -> 逆变换
    U_in_freq = torch.fft.fft2(u_in)
    U_out_freq = U_in_freq * H_transfer
    u_out = torch.fft.ifft2(U_out_freq)
    
    return u_out

# =================模型定义=================
class DiffractiveLayer(nn.Module):
    def __init__(self, h, w, device):
        super().__init__()
        # 透射率 T = A * exp(i*phi)
        self.amplitude = nn.Parameter(torch.rand(1, 1, h, w, device=device))
        self.phase = nn.Parameter(torch.rand(1, 1, h, w, device=device) * 2 * np.pi)
        
    def forward(self, field):
        T = self.amplitude * torch.exp(1j * self.phase)
        return field * T

class PCDONN(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        # 定义衍射层
        self.layers = nn.ModuleList([
            DiffractiveLayer(config.input_size, config.input_size, device) 
            for _ in range(len(config.layer_distances))
        ])
        
        # 定义探测器区域掩码
        self.detector_mask = self._create_detector_masks(config.input_size, config.num_classes, device)

    def _create_detector_masks(self, size, num_classes, device):
        # 创建 [Num_Classes, 1, H, W] 的掩码
        masks = []
        pixels_per_class = (size * size) // num_classes
        flat_indices = torch.randperm(size * size, device=device)
        
        for i in range(num_classes):
            mask = torch.zeros(1, 1, size, size, device=device)
            indices = flat_indices[i * pixels_per_class : (i + 1) * pixels_per_class]
            rows = indices // size
            cols = indices % size
            mask[0, 0, rows, cols] = 1.0
            masks.append(mask)
        return torch.cat(masks, dim=0) # [Num_Classes, 1, H, W]

    def forward(self, x_amp, device):
        """
        x_amp: [B, 1, H, W] 输入图像的振幅
        返回: [B, Num_Classes] 预测分数
        """
        B = x_amp.shape[0]
        n_modes = self.config.n_modes
        
        # 【关键修复】初始化为 [B, 1, H, W] 以匹配 intensity_m 的维度
        total_intensity = torch.zeros(B, 1, self.config.input_size, self.config.input_size, device=device)
        
        # === 部分相干光核心逻辑：非相干叠加 ===
        for m in range(n_modes):
            # 1. 构建第 m 个相干模式的输入场
            # 给输入振幅加上随机相位，模拟部分相干光源的不同模式实现
            phase_noise = torch.randn_like(x_amp, device=device)
            # 简单的相位扰动，系数 (m+1)*0.5 用于区分不同模式
            E_in = x_amp * torch.exp(1j * phase_noise * (m + 1) * 0.5) 
            
            field = E_in
            
            # 2. 逐层传播
            for i, layer in enumerate(self.layers):
                field = layer(field)
                z = self.config.layer_distances[i]
                field = angular_spectrum_propagator(field, z, self.config.wavelength, 
                                                    self.config.dx, self.config.dy, device)
            
            # 3. 传播到探测面
            field = angular_spectrum_propagator(field, self.config.z_det, self.config.wavelength, 
                                                self.config.dx, self.config.dy, device)
            
            # 4. 计算该模式的强度 [B, 1, H, W] 并累加
            intensity_m = torch.abs(field)**2
            total_intensity += intensity_m
            
        # 平均强度
        total_intensity /= n_modes
        
        # 5. 探测器读数
        # total_intensity: [B, 1, H, W]
        # detector_mask: [Num_Classes, 1, H, W]
        # 我们需要计算每个 class 的 mask 与 intensity 的点积
        # 为了广播，将 total_intensity 视为 [B, 1, H, W], mask 视为 [C, 1, H, W]
        # 结果应该是 [B, C]
        
        # 方法：将 mask 展开为 [C, H*W], intensity 展开为 [B, H*W]
        # 然后做矩阵乘法: [B, HW] @ [HW, C] -> [B, C]
        
        intensity_flat = total_intensity.view(B, -1) # [B, H*W]
        mask_flat = self.detector_mask.view(self.config.num_classes, -1) # [C, H*W]
        
        scores = torch.matmul(intensity_flat, mask_flat.t()) # [B, C]
        
        return scores

# =================训练与评估=================
def train_pc_donnn(model, loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print(f"Training PC-DONN model (l_train={config.l_train*1000}mm)...")
    
    for epoch in range(config.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            # 确保数据在正确的设备上
            data = data.to(device)       # [B, 1, H, W]
            target = target.to(device)   # [B]
            
            optimizer.zero_grad()
            
            output = model(data, device)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%")

        print(f"Epoch {epoch} Finished. Avg Loss: {running_loss/len(loader):.4f}, Acc: {100*correct/total:.2f}%")

def evaluate_and_plot(model, test_loader, config, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data, device)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if len(all_preds) >= 1000:
                break
    
    print("\n--- Sample Predictions ---")
    for i in range(5):
        print(f"True: {all_targets[i]}, Pred: {all_preds[i]}")
        
    # 可视化
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input[:1].to(device)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(sample_input[0, 0].cpu(), cmap='gray')
    plt.title("Input Image (Amplitude)")
    plt.axis('off')
    
    # 手动运行一次前向传播以获取强度图
    with torch.no_grad():
        B = 1
        n_modes = config.n_modes
        total_intensity = torch.zeros(B, 1, config.input_size, config.input_size, device=device)
        
        phase_noise = torch.randn_like(sample_input, device=device)
        E_in = sample_input * torch.exp(1j * phase_noise)
        field = E_in
        
        for i, layer in enumerate(model.layers):
            field = layer(field)
            field = angular_spectrum_propagator(field, config.layer_distances[i], config.wavelength, config.dx, config.dy, device)
        
        field = angular_spectrum_propagator(field, config.z_det, config.wavelength, config.dx, config.dy, device)
        total_intensity = torch.abs(field)**2
        
        plt.subplot(1, 2, 2)
        # 去掉 channel 维度画图 [1, H, W] -> [H, W]
        plt.imshow(total_intensity[0, 0].cpu(), cmap='jet')
        plt.title("Output Intensity Distribution")
        plt.colorbar()
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("pc_donnn_result.png")
    print("Result saved to pc_donnn_result.png")
    plt.show()

def main():
    print("=== 开始部分相干光学衍射神经网络(PC-DONN)仿真 ===")
    device = config.device
    print(f"Using device: {device}")
    
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor() # 输出 [1, H, W], 值域 [0, 1]
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    model = PCDONN(config, device).to(device)
    
    train_pc_donnn(model, train_loader, config, device)
    
    evaluate_and_plot(model, test_loader, config, device)

if __name__ == "__main__":
    main()