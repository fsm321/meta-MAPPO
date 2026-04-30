import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import copy
import torch
# 【关键】：限制 PyTorch 主进程的线程数
torch.set_num_threads(1)
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from collections import deque
from utils.replaybuffer import ReplayBuffer
from algorithms.mappo import MAPPO_Continuous
from algorithms.meta_mappo import Meta_MAPPO_Continuous
from env.MPE_env import MPEEnv
from datetime import datetime
from evaluate_parallel import evaluate_policy
from utils.normalization import Normalization, RewardScaling
import multiprocessing as mp


# ======================================================================
# 多进程环境后台 Worker
# ======================================================================
def worker(remote, parent_remote, env_class, args):
    parent_remote.close()
    env = env_class(args)
    current_task = 0
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                s_next, r, done, info = env.step(data)

                # 确保 info 是字典，防止部分自定义环境报错
                if not isinstance(info, dict):
                    info = {}

                # 无论是否结束，每一步都实时计算当前的存活胜负状态
                world = getattr(env, 'world', None) or getattr(env.env, 'world', None)
                if world is not None:
                    r_a = sum([1 for a in world.agents if a.team == 0 and not a.is_dead])
                    b_a = sum([1 for a in world.agents if a.team == 1 and not a.is_dead])
                    info['r_a'] = r_a
                    info['b_a'] = b_a

                # 自动重置
                if np.all(done):
                    info['terminal_observation'] = s_next

                remote.send((s_next, r, done, info))

            elif cmd == 'reset':
                current_task = data
                s = env.reset(task_id=current_task)
                remote.send(s)

            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space, env.n))

            elif cmd == 'close':
                remote.close()
                break
        except Exception as e:
            print(f"Worker process error: {e}")
            break

class SubprocMPEVecEnv:
    def __init__(self, num_envs, env_class, args):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=worker, args=(work_remote, remote, env_class, args))
            p.daemon = True
            p.start()
            self.processes.append(p)
        for work_remote in self.work_remotes:
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space, self.n = self.remotes[0].recv()

    def reset(self, task_ids):
        for remote, task_id in zip(self.remotes, task_ids):
            remote.send(('reset', task_id))
        return np.stack([remote.recv() for remote in self.remotes])

    def step(self, actions):
        for i, remote in enumerate(self.remotes):
            remote.send(('step', actions[i]))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def force_reset(self, env_idx, task_id):
        self.remotes[env_idx].send(('reset', task_id))
        return self.remotes[env_idx].recv()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()


# ======================================================================
# 主训练循环
# ======================================================================
def select_train_task(args, total_steps):
    if hasattr(args, "fixed_task") and args.fixed_task in [0, 1, 2]:
        return args.fixed_task

    episode_idx = total_steps // args.max_episode_steps
    if episode_idx < 20000:
        return 0
    if episode_idx < 50000:
        return np.random.choice([0, 1])
    return np.random.choice([0, 1, 2])


def save_norm_stats(args, state_norm, ckpt_idx):
    if not args.use_state_norm or state_norm is None:
        return

    save_path = f"{args.save_dir}/{args.date}/model/{ckpt_idx}"
    os.makedirs(save_path, exist_ok=True)
    np.save(f"{save_path}/norm_mean.npy", state_norm.running_ms.mean)
    np.save(f"{save_path}/norm_std.npy", state_norm.running_ms.std)


def make_buffer_args(args, buffer_size):
    # Meta-MAPPO 的 support/query buffer 需要独立容量，避免共用原始大 buffer。
    new_args = copy.copy(args)
    new_args.buffer_size = buffer_size
    return new_args


def init_meta_task_ids(args, total_steps):
    """
    前半环境作为 support，后半环境作为 query。
    query 尽量复用对应 support 的任务，保证采样分布一致。
    """
    task_ids = np.zeros(args.num_envs, dtype=int)
    support_envs = args.meta_support_envs
    query_envs = args.num_envs - support_envs

    for i in range(support_envs):
        task_ids[i] = select_train_task(args, total_steps)

    for i in range(query_envs):
        paired_support_idx = i % support_envs
        task_ids[support_envs + i] = task_ids[paired_support_idx]

    return task_ids


def select_meta_reset_task(args, env_idx, env_task_ids, total_steps):
    """
    support 环境重置时重新采样任务；
    query 环境重置时跟随对应 support 环境的任务。
    """
    support_envs = args.meta_support_envs

    if env_idx < support_envs:
        return select_train_task(args, total_steps)

    paired_support_idx = (env_idx - support_envs) % support_envs
    return env_task_ids[paired_support_idx]


def main(args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"==> 正在启动 {args.num_envs} 个并行环境...")
    envs = SubprocMPEVecEnv(args.num_envs, MPEEnv, args)
    args.state_dim, args.action_dim, args.max_action = envs.observation_space[0].shape[0], envs.action_space[0].shape[0], float(envs.action_space[0].high[0])

    log_dir = f"{args.save_dir}/{args.date}/logs/{args.algo_name}_parallel_seed{seed}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    shared_agent = (Meta_MAPPO_Continuous(args) if args.algo_name == "Meta-MAPPO" else MAPPO_Continuous(args))
    if args.algo_name == "Meta-MAPPO":
        meta_buffer_args = make_buffer_args(args, args.meta_buffer_size)
        support_buffer = ReplayBuffer(meta_buffer_args)
        query_buffer = ReplayBuffer(meta_buffer_args)
        shared_buffer = None
    else:
        shared_buffer = ReplayBuffer(args)
        support_buffer = None
        query_buffer = None
    if args.restore: shared_agent.restore(0)

    total_steps, total_episodes = 0, 0
    win_history = deque(maxlen=100)

    # 三个任务各自维护一个近期胜率窗口
    task_win_rates = {
        0: deque(maxlen=50),
        1: deque(maxlen=50),
        2: deque(maxlen=50),
    }

    state_norm = Normalization(shape=args.state_dim)
    reward_scaling = RewardScaling(shape=(args.num_envs, 2), gamma=args.gamma)
    if args.algo_name == "Meta-MAPPO":
        env_task_ids = init_meta_task_ids(args, total_steps)
    else:
        env_task_ids = np.array([select_train_task(args, total_steps) for _ in range(args.num_envs)])
    s = envs.reset(env_task_ids)
    episode_steps = np.zeros(args.num_envs, dtype=int)
    red_ids = [0, 1]

    # 防止 save/eval 在同一里程碑被重复触发
    last_save_index = -1
    last_eval_index = -1

    print("==> 开始多进程并行训练...")
    try:
        while total_steps < args.max_train_steps:
            # ==========================================================
            # 1. 先检查 buffer 是否快满；如果快满，先更新网络，再采样新动作
            # ==========================================================
            if args.algo_name == "Meta-MAPPO":
                support_group_size = args.meta_support_envs * len(red_ids)
                query_group_size = (args.num_envs - args.meta_support_envs) * len(red_ids)

                support_add_size = support_group_size
                query_add_size = query_group_size

                buffer_ready = (
                    support_buffer.count + support_add_size > args.meta_buffer_size or
                    query_buffer.count + query_add_size > args.meta_buffer_size
                )

                if buffer_ready:
                    progress = min(total_steps / args.max_train_steps, 1.0)
                    current_performance = sum(win_history) / len(win_history) if len(win_history) > 0 else 0.0

                    if current_performance < 0.35:
                        meta_lr = 0.08 * (1.0 - progress ** 0.5)
                    elif current_performance > 0.70:
                        meta_lr = 0.05 * (1.0 - progress ** 1.5)
                    else:
                        meta_lr = 0.06 * (1.0 - progress)

                    meta_lr = max(meta_lr, 1e-4)

                    sl_a, sl_c, ql_a, ql_c = shared_agent.meta_train_step(
                        support_buffer=support_buffer,
                        query_buffer=query_buffer,
                        total_steps=total_steps,
                        meta_lr=meta_lr,
                        support_group_size=support_group_size,
                        query_group_size=query_group_size,
                        inner_epochs=args.meta_inner_epochs,
                        outer_epochs=args.meta_outer_epochs
                    )

                    writer.add_scalar("Training/Support_Actor_Loss", sl_a, total_steps)
                    writer.add_scalar("Training/Support_Critic_Loss", sl_c, total_steps)
                    writer.add_scalar("Training/Query_Actor_Loss", ql_a, total_steps)
                    writer.add_scalar("Training/Query_Critic_Loss", ql_c, total_steps)

                    # 保留原有 Actor/Critic 图，便于直接和旧实验对比。
                    writer.add_scalar("Training/Actor_Loss", ql_a, total_steps)
                    writer.add_scalar("Training/Critic_Loss", ql_c, total_steps)
                    writer.add_scalar("Training/Meta_LR", meta_lr, total_steps)

                    support_buffer.count = 0
                    query_buffer.count = 0

            else:
                rollout_add_size = args.num_envs * len(red_ids)

                if shared_buffer.count + rollout_add_size > args.buffer_size:
                    al, cl = shared_agent.update(shared_buffer, total_steps)

                    writer.add_scalar("Training/Actor_Loss", al, total_steps)
                    writer.add_scalar("Training/Critic_Loss", cl, total_steps)

                    shared_buffer.count = 0

            # ==========================================================
            # 2. 批量归一化所有并行环境中的红方观测
            # ==========================================================
            s_normed = np.zeros((args.num_envs, len(red_ids), args.state_dim))

            for env_idx in range(args.num_envs):
                for j, rid in enumerate(red_ids):
                    if args.use_state_norm:
                        s_normed[env_idx][j] = state_norm(s[env_idx][rid])
                    else:
                        s_normed[env_idx][j] = s[env_idx][rid]

            s_flat = s_normed.reshape(args.num_envs * len(red_ids), args.state_dim)

            # ==========================================================
            # 3. 共享策略批量选择动作
            # ==========================================================
            a_flat, a_logp_flat = shared_agent.choose_action(s_flat)

            a_batch = a_flat.reshape(args.num_envs, len(red_ids), args.action_dim)
            a_logp_batch = a_logp_flat.reshape(args.num_envs, len(red_ids), 1)

            actions = []

            for env_idx in range(args.num_envs):
                env_actions = [np.zeros(args.action_dim) for _ in range(envs.n)]

                for j, rid in enumerate(red_ids):
                    if args.policy_dist == "Beta":
                        env_actions[rid] = 2 * (a_batch[env_idx][j] - 0.5) * args.max_action
                    else:
                        env_actions[rid] = a_batch[env_idx][j]

                actions.append(env_actions)

            # ==========================================================
            # 4. 并行环境交互
            # ==========================================================
            s_next, r, done, infos = envs.step(actions)
            episode_steps += 1

            # ==========================================================
            # 5. 奖励缩放
            # ==========================================================
            if args.use_reward_scaling:
                raw_red_rewards = np.array([
                    [r[env_idx][rid] for rid in red_ids]
                    for env_idx in range(args.num_envs)
                ])
                scaled_red_rewards = reward_scaling(raw_red_rewards)
            else:
                scaled_red_rewards = np.array([
                    [r[env_idx][rid] * 0.1 for rid in red_ids]
                    for env_idx in range(args.num_envs)
                ])

            # ==========================================================
            # 6. 存储所有环境的红方经验，并处理结束环境
            # ==========================================================
            for env_idx in range(args.num_envs):
                real_s_next = (
                    infos[env_idx]["terminal_observation"]
                    if (np.all(done[env_idx]) and "terminal_observation" in infos[env_idx])
                    else s_next[env_idx]
                )

                for j, rid in enumerate(red_ids):
                    # dw 表示真实终止：死亡/全歼导致结束时不 bootstrap
                    dw = True if done[env_idx][rid] and episode_steps[env_idx] != args.max_episode_steps else False

                    # done_for_gae 表示 GAE 是否断开：时间截断也要断开
                    done_for_gae = done[env_idx][rid] or (episode_steps[env_idx] >= args.max_episode_steps)

                    if args.use_state_norm:
                        s_next_normed = state_norm(real_s_next[rid], update=False)
                    else:
                        s_next_normed = real_s_next[rid]

                    if args.algo_name == "Meta-MAPPO":
                        if env_idx < args.meta_support_envs:
                            target_buffer = support_buffer
                        else:
                            target_buffer = query_buffer
                    else:
                        target_buffer = shared_buffer

                    target_buffer.store(
                        s_normed[env_idx][j],
                        a_batch[env_idx][j],
                        a_logp_batch[env_idx][j],
                        scaled_red_rewards[env_idx][j],
                        s_next_normed,
                        dw,
                        done_for_gae
                    )

                # ======================================================
                # 7. 如果当前环境结束或达到最大步数，统计胜率并重置该环境
                # ======================================================
                if np.all(done[env_idx]) or episode_steps[env_idx] >= args.max_episode_steps:
                    total_episodes += 1

                    if args.use_reward_scaling:
                        reward_scaling.R[env_idx] = np.zeros(2)

                    if "r_a" in infos[env_idx] and "b_a" in infos[env_idx]:
                        r_a = infos[env_idx]["r_a"]
                        b_a = infos[env_idx]["b_a"]

                        if b_a == 0 and r_a > 0:
                            is_win = 1
                        elif episode_steps[env_idx] >= args.max_episode_steps and r_a > b_a:
                            is_win = 1
                        else:
                            is_win = 0

                        win_history.append(is_win)
                        task_win_rates[env_task_ids[env_idx]].append(is_win)

                    if args.algo_name == "Meta-MAPPO":
                        new_task = select_meta_reset_task(args, env_idx, env_task_ids, total_steps)
                    else:
                        new_task = select_train_task(args, total_steps)
                    env_task_ids[env_idx] = new_task
                    s_next[env_idx] = envs.force_reset(env_idx, new_task)
                    episode_steps[env_idx] = 0

                    if total_episodes % 50 == 0 and len(win_history) > 0:
                        writer.add_scalar(
                            "Training/Win_Rate",
                            100 * sum(win_history) / len(win_history),
                            total_episodes
                        )

                        for t_id in range(3):
                            if len(task_win_rates[t_id]) > 0:
                                writer.add_scalar(
                                    f"Training/Task_{t_id}_WinRate",
                                    100 * sum(task_win_rates[t_id]) / len(task_win_rates[t_id]),
                                    total_episodes
                                )

            s = s_next
            total_steps += args.num_envs

            # 保存与评估：只在到达新里程碑时触发一次
            current_episode_index = total_steps // args.max_episode_steps

            if (
                    current_episode_index > 0
                    and current_episode_index % args.save_freq == 0
                    and current_episode_index != last_save_index
            ):
                shared_agent.save(0, total_steps)
                save_norm_stats(args, state_norm, current_episode_index)
                last_save_index = current_episode_index

            if current_episode_index > 0 and current_episode_index % args.evaluate_freq == 0 and current_episode_index != last_eval_index:
                e_r = evaluate_policy(args, MPEEnv(args), [shared_agent, shared_agent, None, None], state_norm)
                writer.add_scalar("eval/reward", e_r, current_episode_index)
                last_eval_index = current_episode_index

        final_ckpt = total_steps // args.max_episode_steps
        shared_agent.save(0, total_steps)
        save_norm_stats(args, state_norm, final_ckpt)
        print(f"Final model saved at checkpoint {final_ckpt}")

    finally:
        envs.close()
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="MAPPO")

    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--meta_buffer_size", type=int, default=3200)
    parser.add_argument("--meta_support_envs", type=int, default=4)
    parser.add_argument("--meta_inner_epochs", type=int, default=1)
    parser.add_argument("--meta_outer_epochs", type=int, default=1)

    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--fixed_task", type=int, default=-1)

    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=int(5e8))
    parser.add_argument("--evaluate_freq", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=1000)

    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--restore", type=bool, default=False)

    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--hidden_width", type=int, default=256)

    parser.add_argument("--buffer_size", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=6400)
    parser.add_argument("--mini_batch_size", type=int, default=1600)

    parser.add_argument("--K_epochs", type=int, default=4)
    parser.add_argument("--lr_a", type=float, default=3e-5)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--entropy_coef", type=float, default=0.03)

    parser.add_argument("--use_adv_norm", type=bool, default=True)
    parser.add_argument("--use_state_norm", type=bool, default=True)
    parser.add_argument("--use_reward_norm", type=bool, default=True)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--set_adam_eps", type=bool, default=True)
    parser.add_argument("--use_tanh", type=bool, default=True)

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.date:
        args.date = datetime.now().strftime("%m%d_%H%M%S")

    main(args, seed=args.seed)
