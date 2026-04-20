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


# ==========================================
# 1. 基础性能评估 (供训练脚本实时调用)
# ==========================================
def evaluate_policy(args, env, agents, state_norm, seed=0, times=10):
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
                # 如果是蓝方 (None)，直接给空动作，让环境回调接管
                if agents[agent_id] is None:
                    actions.append(np.zeros(args.action_dim))
                else:
                    obs = s[agent_id]

                    # 【核心修正】：只使用传进来的 state_norm，千万不要在这里重新初始化或读取硬盘！
                    if args.use_state_norm and state_norm is not None:
                        obs = state_norm(obs, update=False)

                    a, _ = agents[agent_id].choose_action(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)

            s_next, r, done, _ = env.step(actions)
            episode_reward += r[0]
            s = s_next
            dones = done

        evaluate_rewards.append(episode_reward)

    return np.mean(evaluate_rewards)


# ==========================================
# 2. 论文核心图表 1：高阶空战效能评估
# ==========================================
# 注意：为其加上了 state_norm 参数
def evaluate_combat_metrics(args, env, agents, state_norm, times=100):
    total_wins = 0
    total_red_combat_deaths = 0
    total_blue_deaths = 0
    win_steps = []
    total_energy_consumed = []

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
                    obs = s[agent_id]
                    if args.use_state_norm and state_norm is not None:
                        obs = state_norm(obs, update=False)

                    a, _ = agents[agent_id].choose_action(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)
                    episode_energy += np.linalg.norm(action)

            s_next, r, done, info = env.step(actions)
            episode_reward += sum(r[:2])
            s = s_next
            dones = done

        world = getattr(env, 'world', None) or getattr(env.env, 'world', None) or getattr(env.unwrapped, 'world', None)

        if world is not None:
            red_dead = sum([1 for a in world.agents if
                            a.team == 0 and getattr(a, 'is_dead', False) and getattr(a, 'hp', 100) <= 0])
            blue_dead = sum([1 for a in world.agents if a.team == 1 and getattr(a, 'is_dead', False)])
        else:
            red_dead = sum([int(d) for d in dones[:2]])
            blue_dead = 2 if episode_reward > 80 else (1 if episode_reward > 30 else 0)

        total_red_combat_deaths += red_dead
        total_blue_deaths += blue_dead
        total_energy_consumed.append(episode_energy)

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
def evaluate_robustness(args, env, agents, state_norm, noise_std, times=20):
    evaluate_rewards = []
    for i in range(times):
        s = env.reset()
        episode_steps = 0
        episode_reward = 0
        dones = np.zeros(env.n)

        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            actions = []
            noisy_s = [state + np.random.normal(0, noise_std, size=state.shape) for state in s]

            for agent_id in range(env.n):
                if agents[agent_id] is None:
                    actions.append(np.zeros(args.action_dim))
                else:
                    obs = noisy_s[agent_id]
                    if args.use_state_norm and state_norm is not None:
                        obs = state_norm(obs, update=False)

                    a, _ = agents[agent_id].choose_action(obs)
                    action = 2 * (a - 0.5) * args.max_action if args.policy_dist == "Beta" else a
                    actions.append(action)

            s_next, r, done, _ = env.step(actions)
            episode_reward += sum(r[:2])
            s = s_next
            dones = done

        evaluate_rewards.append(episode_reward)
    return np.mean(evaluate_rewards)


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
            elif agent_id == 0 and step >= 50:
                actions.append(np.zeros(args.action_dim))
                s[agent_id] = np.zeros_like(s[agent_id])
            else:
                obs = s[agent_id]
                if args.use_state_norm and state_norm is not None:
                    obs = state_norm(obs, update=False)

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
# 论文数据生成入口脚本 (单独运行此脚本时触发)
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="Meta-MAPPO")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--model_dir", type=str, default="./data/model_to_eval")
    # ... 省略部分 argparse 配置以保持简洁 ...
    parser.add_argument("--use_state_norm", type=bool, default=True)

    args, _ = parser.parse_known_args()
    args.device = torch.device("cpu")

    env = MPEEnv(args)
    args.state_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].shape[0]
    args.max_action = float(env.action_space[0].high[0])

    if args.algo_name == "Meta-MAPPO":
        shared_agent = Meta_MAPPO_Continuous(args)
    else:
        shared_agent = MAPPO_Continuous(args)

    print(f"正在加载 {args.algo_name} 模型...")
    try:
        shared_agent.restore(0)
        print("模型权重加载成功！")
    except Exception as e:
        print(f"模型加载失败，请检查路径。报错信息: {e}")
        exit()

    agents = [shared_agent, shared_agent, None, None]

    # ==========================================
    # 【新增】：在这里（且仅在这里）读取一次硬盘
    # ==========================================
    state_norm = None
    if args.use_state_norm:
        state_norm = Normalization(shape=args.state_dim)
        try:
            state_norm.running_ms.mean = np.load(f"{args.model_dir}/norm_mean.npy")
            state_norm.running_ms.std = np.load(f"{args.model_dir}/norm_std.npy")
            print("==> 状态归一化统计量加载成功！")
        except FileNotFoundError:
            print("==> 警告: 未找到 norm_mean.npy，将使用默认的 0均值 1方差进行评估。")

    print("\n--- 开始进行鲁棒性(抗干扰)测试 ---")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    robustness_rewards = []
    for noise in noise_levels:
        # 注意：把硬盘读出来的 state_norm 传给函数
        r = evaluate_robustness(args, env, agents, state_norm, noise_std=noise, times=20)
        robustness_rewards.append(r)
        print(f"噪声强度 {noise:.1f} -> 平均奖励: {r:.2f}")

    print("\n--- 开始进行失效恢复测试 ---")
    recovery_curve = evaluate_failure_recovery(args, env, agents, state_norm)

    print("\n--- 开始进行 100 局高阶空战效能评估 (实战化指标) ---")
    win_rate, exchange_ratio, avg_win_steps, avg_energy, total_kills, total_deaths = evaluate_combat_metrics(args, env,
                                                                                                             agents,
                                                                                                             state_norm,
                                                                                                             times=100)

    print(f"\n=======================================================")
    print(f"        >>> {args.algo_name} 最终战术效能评估报告 <<<")
    print(f"=======================================================")
    print(f"1. 综合任务胜率:           {win_rate * 100:.1f} %")
    print(f"=======================================================\n")