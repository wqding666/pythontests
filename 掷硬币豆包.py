import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

# ===================== 基础参数设置（可修改） =====================
p = 0.3  # 硬币正面朝上的概率
n = 1000  # 每次试验抛掷次数
repeat_times = 100  # 重复试验的次数

# ===================== (1) 单次试验：相对频率散点图 =====================
# 生成单次试验的结果（0=反面，1=正面）
single_experiment = np.random.binomial(n=1, p=p, size=n)  # 二项分布模拟抛硬币
# 计算累计正面次数
cumulative_heads = np.cumsum(single_experiment)
# 计算相对频率（累计正面次数 / 已抛掷次数）
relative_frequencies = cumulative_heads / np.arange(1, n+1)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(1, n+1), relative_frequencies, alpha=0.6, s=2, color='#2E86AB')
plt.axhline(y=p, color='red', linestyle='--', label=f'理论概率 p={p}')  # 理论概率参考线
plt.xlabel('抛掷次数', fontsize=12)
plt.ylabel('正面朝上的相对频率', fontsize=12)
plt.title(f'单次试验（n={n}次抛掷）：正面朝上相对频率变化', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show(block=False)  # 非阻塞显示，继续后续绘图

# ===================== (2) 重复100次试验：正面次数直方图 =====================
# 重复100次试验，记录每次试验的正面朝上次数
all_heads = []
for _ in range(repeat_times):
    experiment = np.random.binomial(n=1, p=p, size=n)
    heads_count = np.sum(experiment)
    all_heads.append(heads_count)
all_heads = np.array(all_heads)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(all_heads, bins=20, color='#F77F00', alpha=0.7, edgecolor='black')
plt.axvline(x=n*p, color='red', linestyle='--', label=f'理论均值 np={n*p}')  # 理论均值参考线
plt.xlabel('正面朝上的次数', fontsize=12)
plt.ylabel('频次（100次试验中出现的次数）', fontsize=12)
plt.title(f'重复{repeat_times}次试验（每次{n}次抛掷）：正面朝上次数分布', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show(block=False)

# ===================== (3) 计算100次试验的均值，并与np比较 =====================
mean_heads = np.mean(all_heads)
print("="*50)
print(f"100次试验正面朝上次数的平均值：{mean_heads:.2f}")
print(f"理论均值 np = {n} * {p} = {n*p:.2f}")
print(f"均值偏差：{abs(mean_heads - n*p):.2f}（偏差率：{abs(mean_heads - n*p)/(n*p)*100:.2f}%）")
print("="*50)

# ===================== (4) 尝试不同的p和n值（示例：p=0.5, n=500） =====================
def test_different_params(new_p=0.5, new_n=500, new_repeat=100):
    """测试不同p和n的函数"""
    # 重复试验记录正面次数
    new_heads = []
    for _ in range(new_repeat):
        exp = np.random.binomial(n=1, p=new_p, size=new_n)
        new_heads.append(np.sum(exp))
    new_heads = np.array(new_heads)
    new_mean = np.mean(new_heads)
    
    # 输出结果
    print("\n【不同参数测试】")
    print(f"参数：p={new_p}, n={new_n}, 重复{new_repeat}次")
    print(f"试验均值：{new_mean:.2f}，理论均值 np={new_n*new_p:.2f}")
    print(f"偏差率：{abs(new_mean - new_n*new_p)/(new_n*new_p)*100:.2f}%")
    
    # 绘制新参数的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(new_heads, bins=20, color='#06D6A0', alpha=0.7, edgecolor='black')
    plt.axvline(x=new_n*new_p, color='red', linestyle='--', label=f'理论均值 np={new_n*new_p}')
    plt.xlabel('正面朝上的次数', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.title(f'p={new_p}, n={new_n}：{new_repeat}次试验正面次数分布', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 调用函数测试不同参数（可自行修改new_p/new_n）
test_different_params(new_p=0.5, new_n=500)
test_different_params(new_p=0.7, new_n=2000)

# 最后阻塞，保留所有图形
plt.show()