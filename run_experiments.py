import os
import subprocess
from multiprocessing import Pool
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def run_task(cmd):
    print(f"🚀 启动并行任务: {cmd}")
    # 执行命令，将输出重定向到空，防止多个进程在终端里疯狂打印导致卡顿
    subprocess.run(cmd, shell=True)
    print(f"✅ 任务完成: {cmd}")


if __name__ == '__main__':
    algorithms = ["MAPPO", "Meta-MAPPO"]
    all_seeds = [10, 20, 30]

    # ==========================================
    # ⚠️ 核心设置：并行进程数
    # 设定为 2: 运行 seed 10 (2个任务)
    # 设定为 4: 运行 seed 10, 20 (4个任务)
    # 设定为 6: 运行 seed 10, 20, 30 (6个任务)
    # ==========================================
    num_parallel_tasks = 2  # <--- 以后你只需要修改这个数字！

    num_seeds_to_use = num_parallel_tasks // len(algorithms)
    active_seeds = all_seeds[:num_seeds_to_use]
    run_time = datetime.now().strftime("%m%d_%H%M%S")

    cmds = []
    for algo in algorithms:
        for seed in active_seeds:
            unique_date_tag = f"{algo}_seed{seed}_{run_time}"
            cmd = f"python train.py --algo_name {algo} --seed {seed} --date {unique_date_tag}"
            cmds.append(cmd)

    print(f"准备执行 {len(cmds)} 个实验...")
    print(f"选中的算法: {algorithms}")
    print(f"选中的 Seed: {active_seeds}")
    print(f"并行进程数: {num_parallel_tasks}")
    print("-" * 40)

    # 使用进程池开启并行训练
    with Pool(num_parallel_tasks) as p:
        p.map(run_task, cmds)

    print("🎉 所有实验全部训练完毕！你可以去画图了！")