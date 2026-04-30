import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import torch
import numpy as np
import pyglet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env.MPE_env import MPEEnv
from utils.normalization import Normalization
from algorithms.mappo import MAPPO_Continuous
from algorithms.meta_mappo import Meta_MAPPO_Continuous
import argparse
import json

def normalize_obs(args, obs, state_norm):
    if args.use_state_norm and state_norm is not None:
        return state_norm(obs, update=False)
    return obs
# ==========================================
# 1. 基础性能评估
# ==========================================
def evaluate_policy(args, env, agents, state_norm, seed=0, times=100):
    evaluate_rewards = []
    for i in range(times):
        s = env.reset()
        episode_steps = 0
        episode_reward = 0
        dones = np.zeros(env.n)

        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            actions = []

            for agent_id in range(env.n):
                # 【核心修正】：如果是蓝方 (None)，直接给空动作，让环境回调接管
                if agents[agent_id] is None:
                    actions.append(np.zeros(args.action_dim))
                else:
                    # Reuse the training-time normalization statistics during evaluation.
                    obs = normalize_obs(args, s[agent_id], state_norm)
                    # 【核心修正】：使用 choose_action 并只取第一个返回值 (动作)
                    a, _ = agents[agent_id].choose_action(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)

            s_next, r, done, _ = env.step(actions)
            # 评估时只计算红方的 reward 总和
            episode_reward += sum(r[:2])

            s = s_next
            dones = done

        evaluate_rewards.append(episode_reward)
    return np.mean(evaluate_rewards)


# ==========================================
# 2. 论文核心图表 1：高阶空战效能评估 (纯净战损比、耗时、能量)
# ==========================================
def evaluate_combat_metrics(args, env, agents, state_norm, times=100):
    total_wins = 0
    total_red_combat_deaths = 0  # 我方【真实被击杀】数
    total_blue_deaths = 0  # 击落敌机数
    win_steps = []  # 获胜局的耗时
    total_energy_consumed = []  # 能量消耗

    for i in range(times):
        s = env.reset()
        episode_steps = 0
        dones = np.zeros(env.n)
        episode_energy = 0.0
        episode_reward = 0.0

        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            actions = []

            for agent_id in range(env.n):
                if agents[agent_id] is None:
                    actions.append(np.zeros(args.action_dim))
                else:
                    obs = normalize_obs(args, s[agent_id], state_norm)
                    a, _ = agents[agent_id].choose_action(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)
                    episode_energy += np.linalg.norm(action) # 记录红方能量

            s_next, r, done, info = env.step(actions)
            episode_reward += sum(r[:2])
            s = s_next
            dones = done

        # --- 回合结束，进行【纯净】战损统计 ---
        world = getattr(env, 'world', None) or getattr(env.env, 'world', None) or getattr(env.unwrapped, 'world', None)

        if world is not None:
            # 红方只有在 hp <= 0 时才算被击杀，越界(hp>0)不计入战损
            red_dead = sum([1 for a in world.agents if a.team == 0 and getattr(a, 'is_dead', False) and getattr(a, 'hp', 100) <= 0])
            blue_dead = sum([1 for a in world.agents if a.team == 1 and getattr(a, 'is_dead', False)])
        else:
            red_dead = sum([int(d) for d in dones[:2]])
            blue_dead = 2 if episode_reward > 80 else (1 if episode_reward > 30 else 0)

        total_red_combat_deaths += red_dead
        total_blue_deaths += blue_dead
        total_energy_consumed.append(episode_energy)

        # 胜利条件：全歼敌机，或时间耗尽时击落数大于我方真实阵亡数
        is_win = (blue_dead == 2) or (blue_dead > red_dead)

        if is_win:
            total_wins += 1
            win_steps.append(episode_steps)

    win_rate = total_wins / times
    exchange_ratio = total_blue_deaths / max(total_red_combat_deaths, 1e-5)
    avg_win_steps = np.mean(win_steps) if len(win_steps) > 0 else args.max_episode_steps
    avg_energy = np.mean(total_energy_consumed)

    return win_rate, exchange_ratio, avg_win_steps, avg_energy, total_blue_deaths, total_red_combat_deaths

# ==========================================
# 3. 论文核心图表 2：抗干扰能力 (鲁棒性) 测试
# ==========================================
def evaluate_robustness(args, env, agents, state_norm, noise_std, times=100):
    """
    观测噪声鲁棒性测试：
    在观测状态中加入高斯噪声 N(0, noise_std)，
    每个噪声强度下测试 times 局，返回平均胜率和平均奖励。
    """
    total_wins = 0
    evaluate_rewards = []

    for i in range(times):
        s = env.reset()
        episode_steps = 0
        episode_reward = 0.0
        dones = np.zeros(env.n)

        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            actions = []

            noisy_s = [
                state + np.random.normal(0, noise_std, size=state.shape)
                for state in s
            ]

            for agent_id in range(env.n):
                if agents[agent_id] is None:
                    actions.append(np.zeros(args.action_dim))
                else:
                    obs = normalize_obs(args, noisy_s[agent_id], state_norm)
                    a, _ = agents[agent_id].choose_action(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)

            s_next, r, done, _ = env.step(actions)
            episode_reward += sum(r[:2])

            s = s_next
            dones = done

        evaluate_rewards.append(episode_reward)

        world = getattr(env, 'world', None) or getattr(env.env, 'world', None) or getattr(env.unwrapped, 'world', None)

        if world is not None:
            red_dead = sum([
                1 for a in world.agents
                if a.team == 0 and getattr(a, 'is_dead', False) and getattr(a, 'hp', 100) <= 0
            ])
            blue_dead = sum([
                1 for a in world.agents
                if a.team == 1 and getattr(a, 'is_dead', False)
            ])
        else:
            red_dead = sum([int(d) for d in dones[:2]])
            blue_dead = 2 if episode_reward > 80 else (1 if episode_reward > 30 else 0)

        # 与 combat_metrics 保持一致：全歼敌方，或击落数大于我方真实阵亡数
        is_win = (blue_dead == 2) or (blue_dead > red_dead)

        if is_win:
            total_wins += 1

    win_rate = total_wins / times * 100.0
    avg_reward = np.mean(evaluate_rewards)

    return win_rate, avg_reward

# ==========================================
# 4. 论文核心图表 3：失效恢复过程动态展示
# ==========================================
def evaluate_failure_recovery(args, env, agents, state_norm):
    s = env.reset()
    step_rewards = []
    dones = np.zeros(env.n)

    for step in range(args.max_episode_steps):
        actions = []

        for agent_id in range(env.n):
            if agents[agent_id] is None:
                actions.append(np.zeros(args.action_dim))
            elif agent_id == 0 and step >= 50:  # 模拟1号机在50步后坠毁或宕机
                actions.append(np.zeros(args.action_dim))
                s[agent_id] = np.zeros_like(s[agent_id])
            else:
                obs = normalize_obs(args, s[agent_id], state_norm)
                a, _ = agents[agent_id].choose_action(obs)
                action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                actions.append(action)

        s_next, r, done, _ = env.step(actions)
        step_rewards.append(sum(r[:2]))
        s = s_next

        if all(done):
            step_rewards.extend([0] * (args.max_episode_steps - len(step_rewards)))
            break

    return step_rewards


# ==========================================
# 论文数据生成入口脚本
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="MAPPO")

    parser.add_argument("--date", type=str, default="eval")
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--max_action", type=float, default=1.0)
    parser.add_argument("--model_dir", type=str, default="./data/model_to_eval")
    parser.add_argument("--policy_dist", type=str, default="Gaussian")

    parser.add_argument("--hidden_width", type=int, default=256)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=5e8)

    parser.add_argument("--batch_size", type=int, default=6000)
    parser.add_argument("--mini_batch_size", type=int, default=1000)

    parser.add_argument("--K_epochs", type=int, default=4)
    parser.add_argument("--lr_a", type=float, default=3e-5)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--entropy_coef", type=float, default=0.03)

    parser.add_argument("--use_adv_norm", type=bool, default=True)
    parser.add_argument("--use_state_norm", type=bool, default=True)
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--set_adam_eps", type=bool, default=True)
    parser.add_argument("--use_tanh", type=bool, default=True)

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 核心执行逻辑
    # ==========================================
    env = MPEEnv(args)
    args.state_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].shape[0]
    args.max_action = float(env.action_space[0].high[0])

    # 【核心修正】：评估时同样要使用共享大脑 + 蓝方规则
    if args.algo_name == "Meta-MAPPO":
        shared_agent = Meta_MAPPO_Continuous(args)
    else:
        shared_agent = MAPPO_Continuous(args)

    print(f"正在加载 {args.algo_name} 模型...")
    try:
        shared_agent.restore(0)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败，请检查路径。报错信息: {e}")
        exit()

    # 包装为环境需要的列表：前两个红方是共享神网，后两个蓝方是规则(None)
    agents = [shared_agent, shared_agent, None, None]

    print("\n--- 开始进行鲁棒性(抗干扰)测试 ---")
    state_norm = None
    if args.use_state_norm:
        state_norm = Normalization(shape=args.state_dim)
        try:
            # Load the exact normalization statistics saved alongside the checkpoint.
            state_norm.running_ms.mean = np.load(f"{args.model_dir}/norm_mean.npy")
            state_norm.running_ms.std = np.load(f"{args.model_dir}/norm_std.npy")
            print("状态归一化统计量加载成功。")
        except FileNotFoundError:
            print("未找到 norm_mean.npy / norm_std.npy，将按原始状态评估。")
            state_norm = None

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    robustness_winrates = []
    robustness_rewards = []

    for noise in noise_levels:
        win_rate, avg_reward = evaluate_robustness(
            args,
            env,
            agents,
            state_norm,
            noise_std=noise,
            times=100
        )

        robustness_winrates.append(win_rate)
        robustness_rewards.append(avg_reward)

        print(
            f"噪声强度 {noise:.1f} -> "
            f"胜率: {win_rate:.1f}% | 平均奖励: {avg_reward:.2f}"
        )

    np.save(f"robustness_winrate_{args.algo_name}.npy", robustness_winrates)
    np.save(f"robustness_reward_{args.algo_name}.npy", robustness_rewards)

    # 保留旧文件名，防止旧版 plot_combined.py 报错
    np.save(f"robustness_data_{args.algo_name}.npy", robustness_rewards)
    print("\n--- 开始进行失效恢复测试 ---")
    recovery_curve = evaluate_failure_recovery(args, env, agents, state_norm)
    np.save(f"recovery_data_{args.algo_name}.npy", recovery_curve)

    print("\n--- 开始进行 100 局高阶空战效能评估 (实战化指标) ---")
    win_rate, exchange_ratio, avg_win_steps, avg_energy, total_kills, total_deaths = evaluate_combat_metrics(
        args, env, agents, state_norm, times=100
    )
    print(f"\n=======================================================")
    print(f"        >>> {args.algo_name} 最终战术效能评估报告 <<<")
    print(f"=======================================================")
    print(f"1. 综合任务胜率 (Win Rate):           {win_rate * 100:.1f} %")
    print(f"2. 战损交换比 (Loss-Exchange Ratio):  {exchange_ratio:.2f} "
          f"(共击落 {total_kills} 架 / 阵亡 {total_deaths} 架)")
    print(f"3. 获胜平均耗时 (Avg Time-to-Kill):   {avg_win_steps:.1f} 步")
    print(f"4. 机动能量消耗 (Maneuver Energy):    {avg_energy:.1f}")
    print(f"=======================================================\n")
    num_eval_episodes = 100
    num_red = 2

    combat_metrics = {
        "algo_name": args.algo_name,
        "num_eval_episodes": num_eval_episodes,
        "avg_kills": total_kills / num_eval_episodes,
        "survival_rate": (num_red * num_eval_episodes - total_deaths) / (num_red * num_eval_episodes) * 100.0,
        "exchange_ratio": exchange_ratio,
        "avg_win_steps": avg_win_steps,
        "avg_energy": avg_energy,
        "total_kills": int(total_kills),
        "total_deaths": int(total_deaths),
        "win_rate": win_rate * 100.0
    }

    with open(f"combat_metrics_{args.algo_name}.json", "w", encoding="utf-8") as f:
        json.dump(combat_metrics, f, ensure_ascii=False, indent=4)

    print(f"战术效能指标已保存: combat_metrics_{args.algo_name}.json")