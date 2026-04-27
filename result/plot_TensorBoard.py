import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_data_from_base_dir(base_dir, keyword):
    if not os.path.exists(base_dir):
        print(f"❌ 路径不存在，跳过: {base_dir}")
        return None, None

    # 遍历该时间文件夹下的所有子文件夹
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                try:
                    ea = EventAccumulator(root)
                    ea.Reload()
                    tags = ea.Tags().get('scalars', [])
                    for tag in tags:
                        # 只要标签里包含我们要找的关键字，就立刻提取数据
                        if keyword.lower() in tag.lower():
                            print(f"   ✅ 成功提取 [{keyword}] 数据，来源标签: {tag}")
                            events = ea.Scalars(tag)
                            steps = [e.step for e in events]
                            values = [e.value for e in events]
                            return np.array(steps), np.array(values)
                except Exception as e:
                    continue

    print(f"   ❌ 在 {base_dir} 中未找到包含 '{keyword}' 的数据。")
    return None, None


def plot_metric(algo_logs, keyword, title, ylabel, output_name):
    plt.figure(figsize=(10, 6))
    colors = {"MAPPO": "#4C72B0", "Meta-MAPPO": "#DD8452"}
    has_data = False

    for algo, dirs in algo_logs.items():
        all_steps = []
        all_values = []
        for d in dirs:
            if d.strip().startswith("#") or "你的时间文件夹" in d:
                continue

            steps, vals = get_data_from_base_dir(d, keyword)
            if vals is not None:
                all_steps.append(steps)
                all_values.append(vals)

        if all_values:
            has_data = True
            # 自动对齐最短的数据长度（防止某个seed意外中断报错）
            min_len = min([len(v) for v in all_values])
            all_values_clipped = [v[:min_len] for v in all_values]
            steps_clipped = all_steps[0][:min_len]

            # 计算多seed均值和方差阴影
            mean_vals = np.mean(all_values_clipped, axis=0)
            std_vals = np.std(all_values_clipped, axis=0)

            plt.plot(steps_clipped, mean_vals, label=algo, linewidth=2, color=colors[algo])
            plt.fill_between(steps_clipped, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=colors[algo])

    if has_data:
        plt.xlabel('Logged Step', fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        plt.savefig(output_name, dpi=300)
        print(f"🎉 成功生成并保存图表: {output_name}\n")
    else:
        print(f"⚠️ 无法生成 {output_name}，因为没有找到相关数据。\n")
    plt.close()


if __name__ == '__main__':
    sns.set_theme(style="darkgrid")

    # ==========================================
    # 填入你的基础路径（只需填到时间戳文件夹即可）
    # ==========================================
    algo_logs = {
        "MAPPO": [
            r"D:\Meta-MAPPO\Meta-MAPPO\6.4\data\train\MAPPO_Gaussian_seed10\MAPPO_seed10_0320_135440",
            #r"",
            #r""
        ],
        "Meta-MAPPO": [
            r"D:\Meta-MAPPO\Meta-MAPPO\6.4\data\train\Meta-MAPPO_Gaussian_seed10\Meta-MAPPO_seed10_0320_135440",
            #r"D:\Meta-MAPPO\Meta-MAPPO\5\data\train\Meta-MAPPO_Gaussian_seed20\0315_1351",
            #r"D:\Meta-MAPPO\Meta-MAPPO\5\data\train\Meta-MAPPO_Gaussian_seed30\0315_1351"
        ]
    }

    print("====== 1. 开始处理【奖励曲线】 ======")
    plot_metric(algo_logs, keyword="eval/reward",
                title="Evaluation Reward Comparison",
                ylabel="Evaluation Reward",
                output_name="1_Reward_Curve.png")

    print("====== 2. 开始处理【胜率曲线】 ======")
    plot_metric(
        algo_logs,
        keyword="Win_Rate",
        title="Win Rate Comparison",
        ylabel="Win Rate (%)",
        output_name="2_Win_Rate_Curve.png"
    )
