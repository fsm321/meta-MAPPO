import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
def main(args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"==> 正在启动 {args.num_envs} 个并行环境...")
    envs = SubprocMPEVecEnv(args.num_envs, MPEEnv, args)
    args.state_dim, args.action_dim, args.max_action = envs.observation_space[0].shape[0], envs.action_space[0].shape[0], float(envs.action_space[0].high[0])

    log_dir = f"{args.save_dir}/train_parallel/{args.algo_name}_seed{seed}/{args.date}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    shared_agent = (Meta_MAPPO_Continuous(args) if args.algo_name == "Meta-MAPPO" else MAPPO_Continuous(args))
    shared_buffer = ReplayBuffer(args)
    if args.restore: shared_agent.restore(0)

    total_steps, total_episodes = 0, 0
    win_history = deque(maxlen=100)

    # 三个任务各自维护一个近期胜率窗口
    task_win_rates = {
        0: deque(maxlen=50),
        1: deque(maxlen=50),
        2: deque(maxlen=50),
    }

    current_meta_task = np.random.randint(0, 3)

    state_norm = Normalization(shape=args.state_dim)
    reward_scaling = RewardScaling(shape=(args.num_envs, 2), gamma=args.gamma)

    s = envs.reset([current_meta_task] * args.num_envs)
    episode_steps = np.zeros(args.num_envs, dtype=int)
    red_ids = [0, 1]

    # 防止 save/eval 在同一里程碑被重复触发
    last_save_index = -1
    last_eval_index = -1

    print("==> 开始多进程并行训练...")
    try:
        while total_steps < args.max_train_steps:
            s_normed = np.zeros((args.num_envs, len(red_ids), args.state_dim))
            for env_idx in range(args.num_envs):
                for j, rid in enumerate(red_ids):
                    s_normed[env_idx][j] = state_norm(s[env_idx][rid]) if args.use_state_norm else s[env_idx][rid]

            s_flat = s_normed.reshape(args.num_envs * len(red_ids), args.state_dim)
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
            s_next, r, done, infos = envs.step(actions)
            episode_steps += 1

            if args.use_reward_scaling:
                raw_red_rewards = np.array([[r[env_idx][rid] for rid in red_ids] for env_idx in range(args.num_envs)])
                scaled_red_rewards = reward_scaling(raw_red_rewards)
            else:
                scaled_red_rewards = np.array([[r[env_idx][rid] * 0.1 for rid in red_ids] for env_idx in range(args.num_envs)])

            for env_idx in range(args.num_envs):
                real_s_next = infos[env_idx]['terminal_observation'] if (np.all(done[env_idx]) and 'terminal_observation' in infos[env_idx]) else s_next[env_idx]

                for j, rid in enumerate(red_ids):
                    dw = True if done[env_idx][rid] and episode_steps[env_idx] != args.max_episode_steps else False
                    s_next_normed = state_norm(real_s_next[rid], update=False) if args.use_state_norm else real_s_next[rid]
                    shared_buffer.store(s_normed[env_idx][j], a_batch[env_idx][j], a_logp_batch[env_idx][j], scaled_red_rewards[env_idx][j], s_next_normed, dw, done[env_idx][rid])

                # 检查环境是否结束或截断
                if np.all(done[env_idx]) or episode_steps[env_idx] >= args.max_episode_steps:
                    total_episodes += 1

                    # 单独清零当前结束环境的奖励累积器（防止跨回合污染）
                    if args.use_reward_scaling and hasattr(reward_scaling, 'ret'):
                        reward_scaling.ret[env_idx] = np.zeros(2)

                    if 'r_a' in infos[env_idx] and 'b_a' in infos[env_idx]:
                        r_a = infos[env_idx]['r_a']
                        b_a = infos[env_idx]['b_a']

                        # 宽松胜率判定：全歼敌方，或时间耗尽时我方存活数更多
                        if b_a == 0 and r_a > 0:
                            is_win = 1
                        elif episode_steps[env_idx] >= args.max_episode_steps and r_a > b_a:
                            is_win = 1
                        else:
                            is_win = 0

                        win_history.append(is_win)
                        task_win_rates[current_meta_task].append(is_win)

                    # 如果是截断而非自然 done，强制重置该环境
                    if not np.all(done[env_idx]):
                        s_next[env_idx] = envs.force_reset(env_idx, current_meta_task)

                    episode_steps[env_idx] = 0

                    # 每 50 局记录一次全局与分任务胜率
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

            # 网络更新
            if shared_buffer.count >= args.buffer_size:
                progress = min(total_steps / args.max_train_steps, 1.0)
                current_performance = sum(win_history) / len(win_history) if len(win_history) > 0 else 0.0

                # 动态 Meta-LR
                if current_performance < 0.35:
                    meta_lr = 0.1 * (1.0 - progress ** 0.5)
                elif current_performance > 0.70:
                    meta_lr = 0.1 * (1.0 - progress ** 1.5)
                else:
                    meta_lr = 0.1 * (1.0 - progress)

                meta_lr = max(meta_lr, 1e-4)

                if args.algo_name == "Meta-MAPPO":
                    old_w = shared_agent.get_weights()
                    al, cl = shared_agent.update(shared_buffer, total_steps)
                    shared_agent.meta_update(old_w, meta_lr)
                else:
                    al, cl = shared_agent.update(shared_buffer, total_steps)

                writer.add_scalar("Training/Actor_Loss", al, total_steps)
                writer.add_scalar("Training/Critic_Loss", cl, total_steps)
                if args.algo_name == "Meta-MAPPO":
                    writer.add_scalar("Training/Meta_LR", meta_lr, total_steps)

                shared_buffer.count = 0

                # 困难任务优先采样：胜率越低，下一轮被采样概率越高
                task_probs = []
                for t_id in range(3):
                    t_perf = sum(task_win_rates[t_id]) / len(task_win_rates[t_id]) if len(task_win_rates[t_id]) > 0 else 0.5
                    weight = (1.0 - t_perf) + 0.3
                    task_probs.append(weight)

                task_probs = np.array(task_probs, dtype=np.float32)
                task_probs = np.clip(task_probs, 0.2, None)
                task_probs = task_probs / task_probs.sum()

                current_meta_task = np.random.choice([0, 1, 2], p=task_probs)

                writer.add_scalar("Training/Task_0_Prob", task_probs[0], total_steps)
                writer.add_scalar("Training/Task_1_Prob", task_probs[1], total_steps)
                writer.add_scalar("Training/Task_2_Prob", task_probs[2], total_steps)

            # 保存与评估：只在到达新里程碑时触发一次
            current_episode_index = total_steps // args.max_episode_steps

            if current_episode_index > 0 and current_episode_index % args.save_freq == 0 and current_episode_index != last_save_index:
                shared_agent.save(0, total_steps)

                if args.use_state_norm and hasattr(state_norm, 'running_ms'):
                    save_path = f"{args.save_dir}/train_parallel/{args.algo_name}_seed{seed}/{args.date}/model/{total_steps}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    np.save(f"{save_path}/norm_mean.npy", state_norm.running_ms.mean)
                    np.save(f"{save_path}/norm_std.npy", state_norm.running_ms.std)

                last_save_index = current_episode_index

            if current_episode_index > 0 and current_episode_index % args.evaluate_freq == 0 and current_episode_index != last_eval_index:
                e_r = evaluate_policy(args, MPEEnv(args), [shared_agent, shared_agent, None, None], state_norm)
                writer.add_scalar("eval/reward", e_r, current_episode_index)
                last_eval_index = current_episode_index
        #if (total_steps // args.max_episode_steps) % args.save_freq == 0:
            #shared_agent.save(0, total_steps)
            # 【新增】：同时保存状态归一化的统计量
            #if args.use_state_norm and hasattr(state_norm, 'running_ms'):
                #save_path = f"{args.save_dir}/train_parallel/{args.algo_name}_seed{seed}/{args.date}/model/{total_steps}"
                #if not os.path.exists(save_path): os.makedirs(save_path)
                #np.save(f"{save_path}/norm_mean.npy", state_norm.running_ms.mean)
                #np.save(f"{save_path}/norm_std.npy", state_norm.running_ms.std)
        #if (total_steps // args.max_episode_steps) % args.evaluate_freq == 0:
            # 【关键修复】：将 state_norm 传入，保证评估时观测数据的尺度与训练时一致
            #e_r = evaluate_policy(args, MPEEnv(args), [shared_agent, shared_agent, None, None], state_norm)
            #writer.add_scalar("eval/reward", e_r, total_steps // args.max_episode_steps)
        #if total_episodes % 50 == 0:
            #for t_id in range(3):
                #if len(task_win_rates[t_id]) > 0:
                    #writer.add_scalar(
                        #f"Training/Task_{t_id}_WinRate",
                        #100 * sum(task_win_rates[t_id]) / len(task_win_rates[t_id]),
                        #total_episodes
                    #)
    finally:
        envs.close()
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="MAPPO")

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