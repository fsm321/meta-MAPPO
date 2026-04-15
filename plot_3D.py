import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from env.MPE_env import MPEEnv
from algorithms.mappo import MAPPO_Continuous
from algorithms.meta_mappo import Meta_MAPPO_Continuous
import argparse
from datetime import datetime
# ==========================================
# 核心功能：生成动态 GIF 视频图 (用于答辩 PPT)
# ==========================================
def generate_animation(trajectories, num_entities, save_name="3D_Combat_Animation.gif"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {0: 'red', 1: 'red', 2: 'blue', 3: 'blue', 4: 'gray', 5: 'gray'}
    labels = {0: 'Red UAV 1 (RL)', 1: 'Red UAV 2 (RL)', 2: 'Blue UAV 1 (Rule)', 3: 'Blue UAV 2 (Rule)'}
    line_styles = {0: '-', 1: '--', 2: '-', 3: '--', 4: ':', 5: ':'}

    lines = [ax.plot([], [], [], color=colors.get(i, 'black'), linestyle=line_styles.get(i, '-'),
                     label=labels.get(i, f'UAV {i}'), linewidth=2)[0] for i in range(num_entities)]
    points = [ax.plot([], [], [], color=colors.get(i, 'black'), marker='^', markersize=8)[0] for i in
              range(num_entities)]

    all_x = sum([trajectories[i]["x"] for i in range(num_entities)], [])
    all_y = sum([trajectories[i]["y"] for i in range(num_entities)], [])
    all_z = sum([trajectories[i]["z"] for i in range(num_entities)], [])

    # 加入保险，防止坐标都在同一个点导致报错
    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min, y_max = min(all_y) - 1, max(all_y) + 1
    z_min, z_max = min(all_z) - 0.5, max(all_z) + 0.5
    if x_min == x_max: x_max += 1.0
    if y_min == y_max: y_max += 1.0
    if z_min == z_max: z_max += 1.0

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax.set_xlabel('X Position (km)', fontweight='bold')
    ax.set_ylabel('Y Position (km)', fontweight='bold')
    ax.set_zlabel('Z Altitude (km)', fontweight='bold')
    ax.set_title('3D Air Combat Dynamic Trajectory', fontsize=16, fontweight='bold')
    ax.view_init(elev=30, azim=60)
    plt.legend(loc='upper right')

    def update(frame):
        for i in range(num_entities):
            lines[i].set_data(trajectories[i]["x"][:frame], trajectories[i]["y"][:frame])
            lines[i].set_3d_properties(trajectories[i]["z"][:frame])
            if frame > 0:
                points[i].set_data([trajectories[i]["x"][frame - 1]], [trajectories[i]["y"][frame - 1]])
                points[i].set_3d_properties([trajectories[i]["z"][frame - 1]])
        return lines + points

    total_frames = len(trajectories[0]["x"])
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    ani.save(save_name, writer='pillow', fps=15)
    print(f"🎬 答辩动图已生成: {save_name} (请将其拖入 PPT 中直接播放)")
    plt.close()

# ==========================================
# 补充：运行环境以提取轨迹的逻辑
# ==========================================
def extract_and_plot(args, env, agents):
    # 强制让蓝机执行 task_id=2 (即追击或缠斗战术)，以便生成的图表更具观赏性
    s = env.reset(task_id=2)
    num_entities = len(env.world.agents)
    trajectories = {i: {"x": [], "y": [], "z": []} for i in range(num_entities)}

    print("开始模拟对局并记录轨迹数据...")
    for step in range(args.max_episode_steps):
        actions = []
        for agent_id in range(env.n):
            a = agents[agent_id].evaluate(s[agent_id])
            actions.append(a)

        s_next, r, done, _ = env.step(actions)

        # 记录每架无人机的位置
        for i, agent in enumerate(env.world.agents):
            trajectories[i]["x"].append(agent.state.p_pos[0])
            trajectories[i]["y"].append(agent.state.p_pos[1])
            trajectories[i]["z"].append(agent.state.z_pos)

        s = s_next
        if all(done):
            break

    print("\n数据提取完毕！正在渲染图像 (生成 GIF 需要约几十秒，请耐心等待...)")
    curr_time = datetime.now().strftime("%m%d_%H%M%S")
    gif_name = f"3D_Combat_{args.algo_name}_{curr_time}.gif"
    generate_animation(trajectories, num_entities, save_name=gif_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="Meta-MAPPO")
    parser.add_argument("--model_dir", type=str, default="./data/model_to_eval")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--hidden_width", type=int, default=128)
    parser.add_argument("--use_tanh", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--max_action", type=float, default=1.0)

    # 虚拟参数防报错
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--date", type=str, default="eval")
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=51000000)
    parser.add_argument("--lr_a", type=float, default=1e-4)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--K_epochs", type=int, default=5)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--set_adam_eps", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_adv_norm", type=bool, default=True)

    args = parser.parse_args()
    args.device = torch.device("cpu")

    env = MPEEnv(args)
    args.state_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].shape[0]
    args.max_action = float(env.action_space[0].high[0])

    agents = []
    for agent_id in range(env.n):
        if args.algo_name == "Meta-MAPPO":
            agents.append(Meta_MAPPO_Continuous(args))
        else:
            agents.append(MAPPO_Continuous(args))

    try:
        for agent_id in range(env.n):
            agents[agent_id].restore(agent_id)
        print("✅ 模型加载成功！开始模拟空战轨迹...")
    except Exception as e:
        print(f"❌ 模型加载失败，报错信息: {e}")
        exit()

    extract_and_plot(args, env, agents)