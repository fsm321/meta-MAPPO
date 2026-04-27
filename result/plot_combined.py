import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置学术风格
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


OUTPUT_DIR = "./result"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_combat_metrics(algo_name):
    """
    从 evaluate.py 保存的 json 文件中读取最终作战效能指标。
    文件名格式：
        combat_metrics_MAPPO.json
        combat_metrics_Meta-MAPPO.json
    """
    file_path = f"combat_metrics_{algo_name}.json"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"未找到 {file_path}。\n"
            f"请先运行：python evaluate.py --algo_name {algo_name} --model_dir 你的模型路径"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


def add_bar_labels(ax, bars, values, fmt="{:.2f}"):
    """
    给柱状图添加数值标签。
    """
    max_value = max(values) if max(values) > 0 else 1.0

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max_value * 0.04,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )


def plot_combat_metrics(mappo_metrics, meta_metrics):
    """
    最终作战效能对比图。

    包含 5 个指标：
    1. 平均击落数
    2. 己方存活率
    3. 战损交换比
    4. 平均获胜步数
    5. 机动能量消耗
    """
    labels = ["MAPPO", "Meta-MAPPO"]
    colors = ["#4C72B0", "#DD8452"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    metric_configs = [
        {
            "key": "avg_kills",
            "title": "平均击落数",
            "ylabel": "Destroyed Blue UAVs / Episode",
            "fmt": "{:.2f}",
        },
        {
            "key": "survival_rate",
            "title": "己方存活率",
            "ylabel": "Survival Rate (%)",
            "fmt": "{:.1f}%",
        },
        {
            "key": "exchange_ratio",
            "title": "战损交换比",
            "ylabel": "Kills / Losses",
            "fmt": "{:.2f}",
        },
        {
            "key": "avg_win_steps",
            "title": "平均获胜步数",
            "ylabel": "Time Steps",
            "fmt": "{:.1f}",
        },
        {
            "key": "avg_energy",
            "title": "机动能量消耗",
            "ylabel": "Action L2 Norm",
            "fmt": "{:.1f}",
        },
    ]

    for idx, cfg in enumerate(metric_configs):
        ax = axes[idx]

        values = [
            mappo_metrics[cfg["key"]],
            meta_metrics[cfg["key"]]
        ]

        bars = ax.bar(
            labels,
            values,
            color=colors,
            width=0.52,
            edgecolor="black",
            linewidth=1.3
        )

        ax.set_title(cfg["title"], fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(cfg["ylabel"], fontsize=12)

        upper = max(values) * 1.25 if max(values) > 0 else 1.0
        ax.set_ylim(0, upper)

        add_bar_labels(ax, bars, values, fmt=cfg["fmt"])

    # 第 6 个子图用于文字说明
    axes[5].axis("off")
    summary_text = (
        "指标说明：\n"
        "平均击落数：每局平均击落蓝方无人机数量\n"
        "己方存活率：红方最终存活数量占初始数量比例\n"
        "战损交换比：击落敌机数量 / 己方损失数量\n"
        "平均获胜步数：获胜回合平均结束步数\n"
        "机动能量消耗：红方动作幅值累计 L2 范数"
    )

    axes[5].text(
        0.02,
        0.85,
        summary_text,
        fontsize=12,
        va="top",
        ha="left",
        linespacing=1.8
    )

    plt.suptitle(
        "MAPPO 与 Meta-MAPPO 最终作战效能对比",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "combined_combat_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"已生成: {save_path}")


def plot_combined_robustness():
    """
    抗干扰能力测试对比：
    横轴为观测噪声标准差，纵轴为每个噪声强度下的平均胜率。
    """
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    try:
        mappo_data = np.load("robustness_winrate_MAPPO.npy")
        meta_data = np.load("robustness_winrate_Meta-MAPPO.npy")
    except FileNotFoundError:
        print("未找到 robustness_winrate_xxx.npy，跳过鲁棒性胜率图。")
        print("请先运行 evaluate.py 生成噪声鲁棒性胜率数据。")
        return

    plt.figure(figsize=(8, 6))

    plt.plot(
        noise_levels,
        mappo_data,
        marker="s",
        markersize=8,
        linewidth=2.5,
        color="#4C72B0",
        label="MAPPO"
    )

    plt.plot(
        noise_levels,
        meta_data,
        marker="o",
        markersize=8,
        linewidth=2.5,
        color="#DD8452",
        label="Meta-MAPPO"
    )

    plt.xlabel("观测噪声标准差", fontsize=13)
    plt.ylabel("平均胜率 (%)", fontsize=13)
    plt.title("不同观测噪声强度下的胜率对比", fontsize=15, fontweight="bold", pad=15)
    plt.ylim(0, 100)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "combined_robustness_winrate.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"已生成: {save_path}")


def plot_combined_recovery():
    """
    失效恢复动态过程对比。
    默认读取 evaluate.py 生成的：
        recovery_data_MAPPO.npy
        recovery_data_Meta-MAPPO.npy
    """
    try:
        mappo_data = np.load("recovery_data_MAPPO.npy")
        meta_data = np.load("recovery_data_Meta-MAPPO.npy")
    except FileNotFoundError:
        print("未找到 recovery_data_xxx.npy，跳过失效恢复图。")
        return

    steps = range(len(mappo_data))

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        mappo_data,
        linewidth=2.5,
        color="#4C72B0",
        alpha=0.8,
        label="MAPPO"
    )

    plt.plot(
        steps,
        meta_data,
        linewidth=2.5,
        color="#DD8452",
        alpha=0.9,
        label="Meta-MAPPO"
    )

    plt.axvline(
        x=50,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label="无人机失效点 Step=50"
    )

    y_min = min(np.min(mappo_data), np.min(meta_data))
    plt.text(
        52,
        y_min,
        "部分无人机失效\n策略恢复过程",
        color="#C44E52",
        fontsize=11
    )

    plt.xlabel("时间步", fontsize=13)
    plt.ylabel("团队即时奖励", fontsize=13)
    plt.title("无人机失效后的战术恢复过程对比", fontsize=15, fontweight="bold", pad=15)
    plt.legend(fontsize=12, loc="best")
    plt.xlim(0, len(mappo_data))
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "combined_recovery.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"已生成: {save_path}")


if __name__ == "__main__":
    try:
        MAPPO_METRICS = load_combat_metrics("MAPPO")
        META_MAPPO_METRICS = load_combat_metrics("Meta-MAPPO")
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    plot_combat_metrics(MAPPO_METRICS, META_MAPPO_METRICS)

    # 这两个图是辅助实验，有数据就画，没有数据就自动跳过
    plot_combined_robustness()
    plot_combined_recovery()