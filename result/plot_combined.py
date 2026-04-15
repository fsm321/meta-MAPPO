import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置学术风格的绘图样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_combat_metrics(mappo_metrics, meta_metrics):
    """
    1. 全新 2x2 组合柱状图：高阶空战效能评估矩阵
    包含：胜率、战损比、耗时、能量消耗
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    labels = ['MAPPO', 'Meta-MAPPO']
    colors = ['#4C72B0', '#DD8452']

    # ================= 1. 综合任务胜率 (%) =================
    ax = axes[0, 0]
    values = [mappo_metrics['win_rate'], meta_metrics['win_rate']]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
    ax.set_title('综合任务胜率 (%)', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 10)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.05,
                f'{bar.get_height()}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # ================= 2. 战损交换比 =================
    ax = axes[0, 1]
    values = [mappo_metrics['exchange_ratio'], meta_metrics['exchange_ratio']]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
    ax.set_title('战损交换比 (击落/阵亡)', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.05,
                f'{bar.get_height()}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # ================= 3. 获胜平均耗时 =================
    ax = axes[1, 0]
    values = [mappo_metrics['time_to_kill'], meta_metrics['time_to_kill']]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
    ax.set_title('获胜平均耗时 (Time Steps)', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, max(values) * 1.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.05,
                f'{bar.get_height()}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # ================= 4. 机动能量消耗 =================
    ax = axes[1, 1]
    values = [mappo_metrics['energy'], meta_metrics['energy']]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
    ax.set_title('机动能量消耗 (L2 Norm)', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, max(values) * 1.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.05,
                f'{bar.get_height()}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.suptitle('基于多维度战术指标的协同空战效能评估', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为主标题留出空间
    plt.savefig('combined_combat_metrics.png', dpi=300)
    print("已生成: combined_combat_metrics.png")


def plot_combined_robustness():
    """
    2. 组合折线图：抗干扰能力测试对比
    """
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    try:
        mappo_data = np.load("robustness_data_MAPPO.npy")
        meta_data = np.load("robustness_data_Meta-MAPPO.npy")
    except FileNotFoundError:
        print("未找到 robustness_data_xxx.npy 文件，请确保 evaluate.py 已经正确保存了数据。")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, mappo_data, marker='s', markersize=8, linewidth=2.5, color='#4C72B0', label='MAPPO')
    plt.plot(noise_levels, meta_data, marker='o', markersize=8, linewidth=2.5, color='#DD8452',
             label='Meta-MAPPO')

    plt.xlabel('观测噪声标准差 (Noise Std)', fontsize=13)
    plt.ylabel('平均回合奖励 (Average Reward)', fontsize=13)
    plt.title('不同噪声干扰下的算法鲁棒性对比', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig('combined_robustness.png', dpi=300)
    print("已生成: combined_robustness.png")


def plot_combined_recovery():
    """
    3. 组合折线图：失效恢复动态过程对比
    """
    try:
        mappo_data = np.load("recovery_data_MAPPO.npy")
        meta_data = np.load("recovery_data_Meta-MAPPO.npy")
    except FileNotFoundError:
        print("未找到 recovery_data_xxx.npy 文件，请确保 evaluate.py 已经正确保存了数据。")
        return

    steps = range(len(mappo_data))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, mappo_data, linewidth=2.5, color='#4C72B0', alpha=0.8, label='MAPPO')
    plt.plot(steps, meta_data, linewidth=2.5, color='#DD8452', alpha=0.9, label='Meta-MAPPO')

    # 画一条红色的虚线代表失效点
    plt.axvline(x=50, color='#C44E52', linestyle='--', linewidth=2, label='无人机失效点 (Step=50)')

    # 增加一个文本框解释
    plt.text(52, min(np.min(mappo_data), np.min(meta_data)), '部分战友失效\n重组编队', color='#C44E52', fontsize=11)

    plt.xlabel('时间步 (Time Step)', fontsize=13)
    plt.ylabel('团队即时协同奖励 (Team Step Reward)', fontsize=13)
    plt.title('无人机失效后的战术重组与效能恢复曲线', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='best')
    plt.xlim(0, len(steps))
    plt.tight_layout()
    plt.savefig('combined_recovery.png', dpi=300)
    print("已生成: combined_recovery.png")


if __name__ == '__main__':
    # ==============================================================
    # 填入你跑终端打印出来的真实数据：
    # 格式：{'win_rate': 胜率, 'exchange_ratio': 战损比, 'time_to_kill': 耗时, 'energy': 能量消耗}
    # ==============================================================

    # 这里以你刚刚提供的测试数据为例：
    MAPPO_METRICS = {
        'win_rate': 41.0 ,
        'exchange_ratio': 0.63,
        'time_to_kill': 78.1,
        'energy': 123.7
    }

    META_MAPPO_METRICS = {
        'win_rate': 84.0,
        'exchange_ratio': 2.12,
        'time_to_kill': 77.1,
        'energy': 128.0
    }

    # 执行画图
    plot_combat_metrics(MAPPO_METRICS, META_MAPPO_METRICS)
    plot_combined_robustness()
    plot_combined_recovery()