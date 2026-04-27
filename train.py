import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch

torch.set_num_threads(2)
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
from utils.normalization import Normalization, RewardScaling  # 新增导入


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


def main(args, seed):
    print(f"Using device: {args.device}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = MPEEnv(args)
    args.state_dim, args.action_dim, args.max_action = env.observation_space[0].shape[0], env.action_space[0].shape[
        0], float(env.action_space[0].high[0])

    log_dir = f"{args.save_dir}/{args.date}/logs/{args.algo_name}_seed{seed}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    shared_agent = (Meta_MAPPO_Continuous(args) if args.algo_name == "Meta-MAPPO" else MAPPO_Continuous(args))
    shared_buffer = ReplayBuffer(args)
    if args.restore: shared_agent.restore(0)

    total_steps, win_history = 0, deque(maxlen=100)
    last_save_index, last_eval_index = -1, -1

    # 实例化归一化工具
    state_norm = Normalization(shape=args.state_dim)
    # 两个红方智能体共用一个shape=2的RewardScaler，防止串味
    reward_scaling = RewardScaling(shape=2, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        current_task = select_train_task(args, total_steps)
        s = env.reset(task_id=current_task)
        episode_steps, dones, episode_rewards = 0, np.zeros(env.n), np.zeros(env.n)
        # 每回合重置 RewardScaling 的累积器
        if args.use_reward_scaling:
            reward_scaling.reset()

        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            red_ids = [0, 1]
            # ==========================================================
            # 先检查 buffer 是否快满；如果快满，先更新网络，再采样新动作
            if shared_buffer.count + len(red_ids) > args.buffer_size:
                progress = min(total_steps / args.max_train_steps, 1.0)
                current_performance = sum(win_history) / len(win_history) if len(win_history) > 0 else 0.0

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
            # ==========================================================
            # 状态归一化，用当前策略选择动作
            if args.use_state_norm:
                s_batch = np.array([state_norm(s[i]) for i in red_ids])
            else:
                s_batch = np.array([s[i] for i in red_ids])

            a_batch, a_logp_batch = shared_agent.choose_action(s_batch)

            actions = [np.zeros(args.action_dim) for _ in range(env.n)]
            actions_logp = [np.zeros(1) for _ in range(env.n)]

            for j, rid in enumerate(red_ids):
                actions[rid] = 2 * (a_batch[j] - 0.5) * args.max_action if args.policy_dist == "Beta" else a_batch[j]
                actions_logp[rid] = a_logp_batch[j]

            # ==========================================================
            #环境交互
            s_next, r, done, _ = env.step(actions)

            # 统一处理红方奖励归一化
            if args.use_reward_scaling:
                scaled_red_rewards = reward_scaling(np.array([r[0], r[1]]))
            else:
                scaled_red_rewards = [r[0] * 0.1, r[1] * 0.1]
            # ==========================================================
            # 存储红方两个智能体的经验
            for j, rid in enumerate(red_ids):
                #dw 表示真实终止：死亡/全歼导致结束时不 bootstrap
                dw = True if done[rid] and episode_steps != args.max_episode_steps else False
                # done_for_gae 表示 GAE 是否断开：时间截断也要断开
                done_for_gae = done[rid] or (episode_steps >= args.max_episode_steps)
                # 获取 s_next 归一化值（不更新均值方差）
                if args.use_state_norm:
                    s_next_normed = state_norm(s_next[rid], update=False)
                else:
                    s_next_normed = s_next[rid]

                shared_buffer.store(s_batch[j], actions[rid], actions_logp[rid], scaled_red_rewards[j], s_next_normed,
                                    dw, done_for_gae)
                episode_rewards[rid] += r[rid]
            s, total_steps = s_next, total_steps + 1

        # 胜率统计
        world = getattr(env, 'world', None) or getattr(env.env, 'world', None)
        if world:
            r_a = sum([1 for a in world.agents if a.team == 0 and not a.is_dead])
            b_a = sum([1 for a in world.agents if a.team == 1 and not a.is_dead])

            # win_history.append(1 if (b_a == 0 and r_a > 0) else 0)
            # 放宽后的逻辑：全歼敌机，或者时间耗尽时我方存活数大于敌方存活数
            if b_a == 0 and r_a > 0:
                is_win = 1
            elif episode_steps == args.max_episode_steps and r_a > b_a:
                is_win = 1
            else:
                is_win = 0
            win_history.append(is_win)
            if (total_steps // args.max_episode_steps) % 50 == 0:
                writer.add_scalar("Training/Win_Rate", 100 * sum(win_history) / len(win_history),
                                  total_steps // args.max_episode_steps)

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
        if (
            current_episode_index > 0
            and current_episode_index % args.evaluate_freq == 0
            and current_episode_index != last_eval_index
        ):
            e_r = evaluate_policy(
                args,
                env,
                [shared_agent, shared_agent, None, None],
                state_norm if args.use_state_norm else None
            )
            writer.add_scalar("eval/reward", e_r, current_episode_index)
            last_eval_index = current_episode_index
    final_ckpt = total_steps // args.max_episode_steps
    shared_agent.save(0, total_steps)
    save_norm_stats(args, state_norm, final_ckpt)
    print(f"Final model saved at checkpoint {final_ckpt}")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default="air_combat_2v2")
    parser.add_argument("--algo_name", type=str, default="MAPPO")  # 建议初期调试先用纯MAPPO

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

    # MAPPO 核心调参修改点
    parser.add_argument("--buffer_size", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=6000)
    parser.add_argument("--mini_batch_size", type=int, default=1000)
    parser.add_argument("--K_epochs", type=int, default=4)  # 原4 -> 2，防止Actor迈步过大
    parser.add_argument("--lr_a", type=float, default=3e-5)  # 原1e-4 -> 5e-5，更平稳的策略更新
    parser.add_argument("--lr_c", type=float, default=3e-4)  # 原3e-4 -> 5e-4，加速价值网络拟合
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--entropy_coef", type=float, default=0.03)  # 原0.02 -> 0.05，增强探索

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
        from datetime import datetime

        args.date = datetime.now().strftime("%m%d_%H%M%S")

    main(args, seed=args.seed)
