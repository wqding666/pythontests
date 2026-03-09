import matplotlib.pyplot as plt
import numpy as np

# 数据（上面提取的数值）
epochs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
l_zernike = np.array([168.5507, 163.8470, 157.1629, 144.8374, 129.2244, 126.0183,
                      120.1339, 115.3825, 108.7454, 106.0038, 103.8458, 98.2389])

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, l_zernike, marker='o', color='#2E86AB', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('L_zernike', fontsize=12)
plt.title('L_zernike Loss Trend During Training', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(epochs)  # 显示所有 epoch
plt.tight_layout()
plt.axis
plt.axis([0, 13, 0, 170])
plt.show(block=False)  # 非阻塞显示，不影响后续操作
plt.show()  # 最后阻塞，保留图形

