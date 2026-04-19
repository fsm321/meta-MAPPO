import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch

# 【关键】：限制 PyTorch 主进程的线程数，防止与多进程底层抢占资源导致死锁
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
from evaluate import evaluate_policy
from utils.normalization import Normalization, RewardScaling
import multiprocessing as mp


# ======================================================================
# 多进程环境后台 Worker：负责在独立进程中运行单独的 MPEEnv
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

                # 【自动重置与胜率统计机制】
                if np.all(done):
                    world = getattr(env, 'world', None) or getattr(env.env, 'world', None)
                    if world:
                        r_a = sum([1 for a in world.agents if a.team == 0 and not a.is_dead])
                        b_a = sum([1 for a in world.agents if a.team == 1 and not a.is_dead])
                        info['win'] = 1 if (b_a == 0 and r_a > 0) else 0

                    # 保存真实的 terminal_observation 供 Buffer 计算 Advantage
                    info['terminal_observation'] = s_next
                    s_next = env.reset(task_id=current_task)

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


# ======================================================================
# 多进程向量化环境 Wrapper：主进程与多个 Worker 通信的桥梁
# ======================================================================
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
# 主训练循环 (适配了 Batch 处理)
# ======================================================================
def main(args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 实例化并行环境
    print(f"==> 正在启动 {args.num_envs} 个并行环境...")
    envs = SubprocMPEVecEnv(args.num_envs, MPEEnv, args)
    args.state_dim, args.action_dim, args.max_action = envs.observation_space[0].shape[0], envs.action_space[0].shape[
        0], float(envs.action_space[0].high[0])

    log_dir = f"{args.save_dir}/train_parallel/{args.algo_name}_seed{seed}/{args.date}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    shared_agent = (Meta_MAPPO_Continuous(args) if args.algo_name == "Meta-MAPPO" else MAPPO_Continuous(args))
    shared_buffer = ReplayBuffer(args)
    if args.restore: shared_agent.restore(0)

    total_steps, win_history = 0, deque(maxlen=100)
    current_meta_task = np.random.randint(0, 3)

    # 初始化归一化器（因为有多个环境，RewardScaling 的 shape 需要对应 [环境数, 红方智能体数]）
    state_norm = Normalization(shape=args.state_dim)
    reward_scaling = RewardScaling(shape=(args.num_envs, 2), gamma=args.gamma)

    # 首次 Reset
    s = envs.reset([current_meta_task] * args.num_envs)
    episode_steps = np.zeros(args.num_envs, dtype=int)
    red_ids = [0, 1]

    print("==> 开始多进程并行训练...")
    while total_steps < args.max_train_steps:
        # 1. 状态归一化并打平 (准备送入 Actor)
        s_normed = np.zeros((args.num_envs, len(red_ids), args.state_dim))
        for env_idx in range(args.num_envs):
            for j, rid in enumerate(red_ids):
                s_normed[env_idx][j] = state_norm(s[env_idx][rid]) if args.use_state_norm else s[env_idx][rid]

        # 将 shape 从 (num_envs, 2, dim) 变为 (num_envs * 2, dim) 以支持 Batch Inference
        s_flat = s_normed.reshape(args.num_envs * len(red_ids), args.state_dim)

        # 2. 批量推理动作
        a_flat, a_logp_flat = shared_agent.choose_action(s_flat)

        # 恢复 shape (num_envs, 2, act_dim)
        a_batch = a_flat.reshape(args.num_envs, len(red_ids), args.action_dim)
        a_logp_batch = a_logp_flat.reshape(args.num_envs, len(red_ids), 1)

        # 3. 构造给每个环境的完整 Actions 列表
        actions = []
        for env_idx in range(args.num_envs):
            env_actions = [np.zeros(args.action_dim) for _ in range(envs.n)]
            for j, rid in enumerate(red_ids):
                env_actions[rid] = 2 * (a_batch[env_idx][j] - 0.5) * args.max_action if args.policy_dist == "Beta" else \
                a_batch[env_idx][j]
            actions.append(env_actions)

        # 4. 环境并行推进
        s_next, r, done, infos = envs.step(actions)
        episode_steps += 1

        # 5. 奖励缩放 (统一处理所有环境的奖励)
        if args.use_reward_scaling:
            raw_red_rewards = np.array([[r[env_idx][rid] for rid in red_ids] for env_idx in range(args.num_envs)])
            scaled_red_rewards = reward_scaling(raw_red_rewards)
        else:
            scaled_red_rewards = np.array(
                [[r[env_idx][rid] * 0.1 for rid in red_ids] for env_idx in range(args.num_envs)])

        # 6. 数据存储与回合维护
        for env_idx in range(args.num_envs):
            # 获取真实的 s_next（防止被 auto-reset 覆盖）
            real_s_next = infos[env_idx]['terminal_observation'] if (
                        np.all(done[env_idx]) and 'terminal_observation' in infos[env_idx]) else s_next[env_idx]

            for j, rid in enumerate(red_ids):
                dw = True if done[env_idx][rid] and episode_steps[env_idx] != args.max_episode_steps else False
                s_next_normed = state_norm(real_s_next[rid], update=False) if args.use_state_norm else real_s_next[rid]

                shared_buffer.store(s_normed[env_idx][j], a_batch[env_idx][j], a_logp_batch[env_idx][j],
                                    scaled_red_rewards[env_idx][j], s_next_normed, dw, done[env_idx][rid])

            # 检查环境是否结束或截断
            if np.all(done[env_idx]) or episode_steps[env_idx] >= args.max_episode_steps:
                # 记录胜率 (由 Worker 后台传回)
                if 'win' in infos[env_idx]:
                    win_history.append(infos[env_idx]['win'])

                # 截断处理：如果没有全灭但步数用尽，必须强制重置该环境
                if not np.all(done[env_idx]):
                    s_next[env_idx] = envs.force_reset(env_idx, current_meta_task)

                episode_steps[env_idx] = 0

        # 写回状态
        s = s_next
        total_steps += args.num_envs  # 走一步相当于单线程走 N 步！

        # 记录胜率
        if (total_steps // args.max_episode_steps) % 50 == 0 and len(win_history) > 0:
            writer.add_scalar("Training/Win_Rate", 100 * sum(win_history) / len(win_history),
                              total_steps // args.max_episode_steps)

        # 7. 网络更新
        if shared_buffer.count >= args.buffer_size:
            meta_lr = 0.1 * (1 - total_steps / args.max_train_steps)
            if args.algo_name == "Meta-MAPPO":
                old_w = shared_agent.get_weights()
                al, cl = shared_agent.update(shared_buffer, total_steps)
                shared_agent.meta_update(old_w, meta_lr)
            else:
                al, cl = shared_agent.update(shared_buffer, total_steps)

            writer.add_scalar("Training/Actor_Loss", al, total_steps)
            writer.add_scalar("Training/Critic_Loss", cl, total_steps)
            shared_buffer.count = 0
            current_meta_task = np.random.randint(0, 3)

        if (total_steps // args.max_episode_steps) % args.save_freq == 0: shared_agent.save(0, total_steps)
        if (total_steps // args.max_episode_steps) % args.evaluate_freq == 0:
            e_r = evaluate_policy(args, MPEEnv(args), [shared_agent, shared_agent, None, None], None)
            writer.add_scalar("eval/reward", e_r, total_steps // args.max_episode_steps)


if __name__ == '__main__':
    # 【必须有此判断】Windows 下多进程 spawn 必须在 main 块内执行
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="MAPPO")  # 默认依然建议先跑通 MAPPO

    # 🌟 新增：多进程环境数量 (建议设为 CPU 核心数的一半，如 8)
    parser.add_argument("--num_envs", type=int, default=8)

    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--date", type=str, default="")

    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=int(5e8))
    parser.add_argument("--evaluate_freq", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=1000)

    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--restore", type=bool, default=False)

    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--hidden_width", type=int, default=256)

    # 与之前一样的优化后 MAPPO 参数
    parser.add_argument("--buffer_size", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=6400)
    parser.add_argument("--mini_batch_size", type=int, default=1600)
    parser.add_argument("--K_epochs", type=int, default=2)
    parser.add_argument("--lr_a", type=float, default=5e-5)
    parser.add_argument("--lr_c", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.05)

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