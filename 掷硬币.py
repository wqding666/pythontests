import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

n=1000 # 抛掷次数
p=0.3  # 正面朝上的概率
trials=np.random.binomial(1,p,n) # 生成n次抛掷的结果（0=反面，1=正面）
cumulative=np.cumsum(trials)     # 计算累计正面次数
relative_frequencies=cumulative / np.arange(1, n+1) # 计算相对频率（累计正面次数 / 已抛掷次数）

#（1） 画出单次试验相对频率散点图
plt.figure(figsize=(8,4))
plt.scatter(np.arange(1,n+1),relative_frequencies,alpha=1,s=0.3)
plt.axis([1, n, 0, 0.6])
plt.axhline(y=p, color='red', linestyle='--', label=f'理论概率 p={p}')  # 理论概率参考线
plt.xlabel('抛掷次数',fontsize=12)
plt.ylabel('正面朝上相对频率',fontsize=12)
plt.title(f'单次试验（n={n}次抛掷）：正面朝上相对频率变化', fontsize=14, fontweight='bold')
plt.show()

#（2） 重复试验100次，画出100次实验正面朝上的直方图
num_experiments=100
success_count=[]
for i in range(num_experiments):
    trials=np.random.binomial(1,p,n)
    success_count.append(np.sum(trials))
plt.figure(figsize=(8,4))
plt.hist(success_count,bins=15,edgecolor='black')
plt.axvline(x=n*p, color='red', linestyle='--', label=f'理论均值 np={n*p}')  # 理论均值参考线
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.xlabel('正面向上次数',fontsize=12)
plt.ylabel('频数')
plt.title(f'重复{num_experiments}次试验（每次{n}次抛掷）：正面朝上次数分布', fontsize=14, fontweight='bold')
#plt.show()

#（3） 计算100次试验的均值，并与np比较
mean_count=np.mean(success_count)
print(f"正面向上次数均值{mean_count:.2f}")
print(f"np={n*p:.2f}")
plt.text(0.95, 0.95, f"均值偏差：{abs(mean_count - n*p):.2f}\n偏差率：{abs(mean_count - n*p)/(n*p)*100:.2f}%",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')

plt.show()

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
    plt.figure(figsize=(8, 4))
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
plt.show()
