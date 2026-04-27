import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from algorithms.mappo import MAPPO_Continuous
from algorithms.meta_mappo import Meta_MAPPO_Continuous
from env.MPE_env import MPEEnv
from utils.normalization import Normalization


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")


def get_world(env):
    """Return the underlying world object across env wrappers."""
    return (
        getattr(env, "world", None)
        or getattr(getattr(env, "env", None), "world", None)
        or getattr(getattr(env, "unwrapped", None), "world", None)
    )


def normalize_obs(args, obs, state_norm):
    """Reuse training-time normalization statistics during visualization."""
    if args.use_state_norm and state_norm is not None:
        return state_norm(obs, update=False)
    return obs


def load_state_norm(args):
    """Load saved normalization statistics if available."""
    if not args.use_state_norm:
        return None

    state_norm = Normalization(shape=args.state_dim)
    mean_path = os.path.join(args.model_dir, "norm_mean.npy")
    std_path = os.path.join(args.model_dir, "norm_std.npy")

    try:
        state_norm.running_ms.mean = np.load(mean_path)
        state_norm.running_ms.std = np.load(std_path)
        print("状态归一化统计量加载成功。")
        return state_norm
    except FileNotFoundError:
        print("未找到 norm_mean.npy / norm_std.npy，将使用原始状态绘制轨迹。")
        return None


def choose_visual_action(agent, obs, args):
    """Choose actions for trajectory visualization."""
    if args.deterministic:
        with torch.no_grad():
            s_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            if len(s_tensor.shape) == 1:
                s_tensor = s_tensor.unsqueeze(0)
            action = agent.actor(s_tensor)
            action = torch.clamp(action, -args.max_action, args.max_action)
        return action.cpu().numpy().flatten()

    action, _ = agent.choose_action(obs)
    if args.policy_dist == "Beta":
        return 2 * (action - 0.5) * args.max_action
    return action


def init_trajectories(num_entities):
    return {
        i: {"x": [], "y": [], "z": [], "dead": [], "hp": []}
        for i in range(num_entities)
    }


def record_positions(world, trajectories):
    """Record positions and combat state for every agent."""
    for i, agent in enumerate(world.agents):
        trajectories[i]["x"].append(agent.state.p_pos[0])
        trajectories[i]["y"].append(agent.state.p_pos[1])
        trajectories[i]["z"].append(agent.state.z_pos)
        trajectories[i]["dead"].append(getattr(agent, "is_dead", False))
        trajectories[i]["hp"].append(getattr(agent, "hp", 100.0))


def is_red_win(world):
    blue_dead = sum(
        1 for agent in world.agents
        if agent.team == 1 and getattr(agent, "is_dead", False)
    )
    red_alive = sum(
        1 for agent in world.agents
        if agent.team == 0 and not getattr(agent, "is_dead", False)
    )
    return blue_dead == 2 and red_alive > 0


def get_battle_score(world):
    blue_dead = sum(
        1 for agent in world.agents
        if agent.team == 1 and getattr(agent, "is_dead", False)
    )
    red_alive = sum(
        1 for agent in world.agents
        if agent.team == 0 and not getattr(agent, "is_dead", False)
    )
    return blue_dead, red_alive


def collect_destroy_events(trajectories):
    """Capture the first frame where each entity becomes dead."""
    destroy_events = {}
    for entity_id, traj in trajectories.items():
        for frame_idx, dead in enumerate(traj["dead"]):
            prev_dead = traj["dead"][frame_idx - 1] if frame_idx > 0 else False
            if dead and not prev_dead:
                destroy_events[entity_id] = {
                    "frame": frame_idx,
                    "x": traj["x"][frame_idx],
                    "y": traj["y"][frame_idx],
                    "z": traj["z"][frame_idx],
                }
                break
    return destroy_events


def detect_pincer(trajectories, frame_idx, blue_id, max_dist=5.0):
    """Detect whether red agents approach the same blue target from both sides."""
    if blue_id not in trajectories or trajectories[blue_id]["dead"][frame_idx]:
        return False

    red0 = np.array([trajectories[0]["x"][frame_idx], trajectories[0]["y"][frame_idx]])
    red1 = np.array([trajectories[1]["x"][frame_idx], trajectories[1]["y"][frame_idx]])
    blue = np.array([trajectories[blue_id]["x"][frame_idx], trajectories[blue_id]["y"][frame_idx]])

    v0 = red0 - blue
    v1 = red1 - blue
    norm0 = np.linalg.norm(v0)
    norm1 = np.linalg.norm(v1)
    if norm0 < 1e-6 or norm1 < 1e-6:
        return False
    if norm0 > max_dist or norm1 > max_dist:
        return False

    cos_angle = np.dot(v0, v1) / (norm0 * norm1)
    return cos_angle < -0.3


def build_frame_title(frame_idx, trajectories, destroy_events, labels, pincer_dist=5.0):
    title = "3D Air Combat Dynamic Trajectory"

    destroy_msgs = []
    for blue_id in (2, 3):
        event = destroy_events.get(blue_id)
        if event is not None and frame_idx >= event["frame"]:
            destroy_msgs.append(f"{labels[blue_id]} Destroyed at Step {event['frame']}")

    if destroy_msgs:
        title += " | " + " | ".join(destroy_msgs)

    if detect_pincer(trajectories, frame_idx, 2, pincer_dist) or detect_pincer(trajectories, frame_idx, 3, pincer_dist):
        title += " | Pincer Attack Detected"

    return title


def simulate_episode(args, env, agents, state_norm):
    """Run one episode and collect a full trajectory."""
    state = env.reset(task_id=args.task_id)
    world = get_world(env)
    if world is None:
        raise RuntimeError("无法获取 env.world，请检查环境封装。")

    num_entities = len(world.agents)
    trajectories = init_trajectories(num_entities)
    record_positions(world, trajectories)

    dones = np.zeros(env.n)
    end_step = args.max_episode_steps

    for step in range(args.max_episode_steps):
        actions = []
        for agent_id in range(env.n):
            if agents[agent_id] is None:
                actions.append(np.zeros(args.action_dim))
            else:
                obs = normalize_obs(args, state[agent_id], state_norm)
                action = choose_visual_action(agents[agent_id], obs, args)
                actions.append(action)

        next_state, reward, done, info = env.step(actions)
        world = get_world(env)
        record_positions(world, trajectories)

        state = next_state
        dones = done
        if np.all(dones):
            end_step = step + 1
            for _ in range(args.post_end_frames):
                record_positions(world, trajectories)
            break

    world = get_world(env)
    blue_dead, red_alive = get_battle_score(world)
    return {
        "trajectories": trajectories,
        "num_entities": num_entities,
        "blue_dead": blue_dead,
        "red_alive": red_alive,
        "red_win": is_red_win(world),
        "end_step": end_step,
    }


def generate_animation(
    trajectories,
    num_entities,
    save_name="3D_Combat_Animation.gif",
    fps=15,
    lock_distance=5.0,
    pincer_dist=5.0
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {0: 'red', 1: 'red', 2: 'blue', 3: 'blue', 4: 'gray', 5: 'gray'}
    labels = {0: 'Red UAV 1', 1: 'Red UAV 2', 2: 'Blue UAV 1', 3: 'Blue UAV 2'}
    line_styles = {0: '-', 1: '--', 2: '-', 3: '--', 4: ':', 5: ':'}

    lines = []
    points = []
    dead_marks = []
    destroy_events = collect_destroy_events(trajectories)

    for entity_id in range(num_entities):
        line = ax.plot(
            [], [], [],
            color=colors.get(entity_id, "black"),
            linestyle=line_styles.get(entity_id, "-"),
            label=labels.get(entity_id, f"UAV {entity_id}"),
            linewidth=2
        )[0]
        point = ax.plot(
            [], [], [],
            color=colors.get(entity_id, "black"),
            marker="^",
            markersize=8
        )[0]
        dead_mark = ax.plot(
            [], [], [],
            marker="x",
            markersize=14,
            color="black",
            linestyle="None"
        )[0]
        lines.append(line)
        points.append(point)
        dead_marks.append(dead_mark)

    attack_lines = [
        ax.plot([], [], [], color="orange", linestyle="--", linewidth=1.5, alpha=0.6)[0]
        for _ in range(2)
    ]

    all_x, all_y, all_z = [], [], []
    for entity_id in range(num_entities):
        all_x.extend(trajectories[entity_id]["x"])
        all_y.extend(trajectories[entity_id]["y"])
        all_z.extend(trajectories[entity_id]["z"])

    if not all_x:
        print("没有轨迹数据，无法生成 GIF。")
        plt.close(fig)
        return

    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min, y_max = min(all_y) - 1, max(all_y) + 1
    z_min, z_max = min(all_z) - 0.5, max(all_z) + 0.5
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 1.0
    if z_min == z_max:
        z_max += 1.0

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel('X Position (km)', fontweight='bold')
    ax.set_ylabel('Y Position (km)', fontweight='bold')
    ax.set_zlabel('Z Altitude (km)', fontweight='bold')
    ax.view_init(elev=30, azim=60)
    plt.legend(loc='upper right')

    total_frames = len(trajectories[0]["x"])
    red_ids = [0, 1]
    blue_ids = [2, 3]

    def update(frame):
        frame_idx = max(frame, 0)

        for entity_id in range(num_entities):
            event = destroy_events.get(entity_id)

            # =========================
            # 蓝方（2,3）被击落后：
            # 轨迹停止增长，位置固定在死亡点
            # =========================
            if entity_id in blue_ids and event is not None and frame_idx >= event["frame"]:
                history_end = event["frame"] + 1
                xs = trajectories[entity_id]["x"][:history_end]
                ys = trajectories[entity_id]["y"][:history_end]
                zs = trajectories[entity_id]["z"][:history_end]

                lines[entity_id].set_data(xs, ys)
                lines[entity_id].set_3d_properties(zs)

                # 当前位置固定在死亡点
                points[entity_id].set_data([event["x"]], [event["y"]])
                points[entity_id].set_3d_properties([event["z"]])

                # 死亡标记固定在死亡点
                dead_marks[entity_id].set_data([event["x"]], [event["y"]])
                dead_marks[entity_id].set_3d_properties([event["z"]])

            else:
                history_end = frame_idx + 1
                xs = trajectories[entity_id]["x"][:history_end]
                ys = trajectories[entity_id]["y"][:history_end]
                zs = trajectories[entity_id]["z"][:history_end]

                lines[entity_id].set_data(xs, ys)
                lines[entity_id].set_3d_properties(zs)

                if len(xs) > 0:
                    points[entity_id].set_data([xs[-1]], [ys[-1]])
                    points[entity_id].set_3d_properties([zs[-1]])

                if event is not None and frame_idx >= event["frame"]:
                    dead_marks[entity_id].set_data([event["x"]], [event["y"]])
                    dead_marks[entity_id].set_3d_properties([event["z"]])
                else:
                    dead_marks[entity_id].set_data([], [])
                    dead_marks[entity_id].set_3d_properties(np.array([]))

        # 红方攻击线：只连接存活蓝方
        for attack_idx, red_id in enumerate(red_ids):
            rx = trajectories[red_id]["x"][frame_idx]
            ry = trajectories[red_id]["y"][frame_idx]
            rz = trajectories[red_id]["z"][frame_idx]

            nearest_blue = None
            nearest_dist = np.inf

            for blue_id in blue_ids:
                # 蓝方死亡后不再参与连线
                if blue_id in destroy_events and frame_idx >= destroy_events[blue_id]["frame"]:
                    continue

                bx = trajectories[blue_id]["x"][frame_idx]
                by = trajectories[blue_id]["y"][frame_idx]
                bz = trajectories[blue_id]["z"][frame_idx]

                dist = np.linalg.norm([rx - bx, ry - by, rz - bz])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_blue = blue_id

            if nearest_blue is not None and nearest_dist < lock_distance:
                bx = trajectories[nearest_blue]["x"][frame_idx]
                by = trajectories[nearest_blue]["y"][frame_idx]
                bz = trajectories[nearest_blue]["z"][frame_idx]
                attack_lines[attack_idx].set_data([rx, bx], [ry, by])
                attack_lines[attack_idx].set_3d_properties([rz, bz])
            else:
                attack_lines[attack_idx].set_data([], [])
                attack_lines[attack_idx].set_3d_properties(np.array([]))

        ax.set_title(
            build_frame_title(frame_idx, trajectories, destroy_events, labels, pincer_dist=pincer_dist),
            fontsize=14,
            fontweight='bold'
        )
        return lines + points + dead_marks + attack_lines
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    ani.save(save_name, writer='pillow', fps=fps)
    print(f"动画已生成: {save_name}")
    plt.close(fig)

def extract_and_plot(args, env, agents, state_norm):
    """Search for a red-win episode first, then render that trajectory."""
    search_count = args.max_search_episodes if args.search_win_episode else 1
    best_episode = None
    best_score = (-1, -1)

    for episode_idx in range(search_count):
        print(f"开始模拟第 {episode_idx + 1}/{search_count} 局，task_id={args.task_id} ...")
        episode_data = simulate_episode(args, env, agents, state_norm)
        blue_dead = episode_data["blue_dead"]
        red_alive = episode_data["red_alive"]

        print(
            f"第 {episode_idx + 1} 局结束: "
            f"blue_dead={blue_dead}, red_alive={red_alive}, end_step={episode_data['end_step']}"
        )

        if episode_data["red_win"]:
            best_episode = episode_data
            print(f"已找到红方全歼蓝方的对局，使用第 {episode_idx + 1} 局生成动画。")
            break

        score = (blue_dead, red_alive)
        if score > best_score:
            best_score = score
            best_episode = episode_data

    if best_episode is None:
        raise RuntimeError("未能采集到任何有效轨迹。")

    if args.search_win_episode and not best_episode["red_win"]:
        print("在搜索范围内未找到红方全歼对局，将使用最接近胜利的一局生成动画。")

    curr_time = datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    gif_name = os.path.join(
        args.output_dir,
        f"3D_Combat_{args.algo_name}_task{args.task_id}_{curr_time}.gif"
    )

    print("轨迹提取完毕，开始生成 GIF ...")
    print(
        f"最终选用轨迹: red_win={best_episode['red_win']}, "
        f"blue_dead={best_episode['blue_dead']}, "
        f"red_alive={best_episode['red_alive']}, "
        f"end_step={best_episode['end_step']}"
    )

    generate_animation(
        best_episode["trajectories"],
        best_episode["num_entities"],
        save_name=gif_name,
        fps=args.gif_fps,
        lock_distance=args.lock_distance,
        pincer_dist=args.lock_distance
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="Meta-MAPPO")
    parser.add_argument("--model_dir", type=str, default="./data/model_to_eval")
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--task_id", type=int, default=2)

    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--hidden_width", type=int, default=256)
    parser.add_argument("--use_tanh", type=str2bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=str2bool, default=True)
    parser.add_argument("--use_state_norm", type=str2bool, default=True)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--search_win_episode", type=str2bool, default=True)
    parser.add_argument("--max_search_episodes", type=int, default=50)
    parser.add_argument("--post_end_frames", type=int, default=80)
    parser.add_argument("--gif_fps", type=int, default=15)
    parser.add_argument("--lock_distance", type=float, default=2.0)
    
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--date", type=str, default="eval")
    parser.add_argument("--batch_size", type=int, default=6000)
    parser.add_argument("--mini_batch_size", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=500000000)
    parser.add_argument("--lr_a", type=float, default=3e-5)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--K_epochs", type=int, default=4)
    parser.add_argument("--entropy_coef", type=float, default=0.03)
    parser.add_argument("--set_adam_eps", type=str2bool, default=True)
    parser.add_argument("--use_grad_clip", type=str2bool, default=True)
    parser.add_argument("--use_lr_decay", type=str2bool, default=True)
    parser.add_argument("--use_adv_norm", type=str2bool, default=True)

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = MPEEnv(args)
    args.state_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].shape[0]
    args.max_action = float(env.action_space[0].high[0])

    if args.algo_name == "Meta-MAPPO":
        shared_agent = Meta_MAPPO_Continuous(args)
    else:
        shared_agent = MAPPO_Continuous(args)

    try:
        shared_agent.restore(0)
        print(f"{args.algo_name} 模型加载成功。")
    except Exception as e:
        print(f"模型加载失败，请检查 model_dir。错误信息: {e}")
        raise SystemExit(1)

    state_norm = load_state_norm(args)
    agents = [shared_agent, shared_agent, None, None]
    extract_and_plot(args, env, agents, state_norm)
