import os
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.v_layer = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.v_layer, gain=1.0)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.v_layer(s)
        return v_s


class MAPPO_Continuous:
    def __init__(self, args):
        self.device = args.device
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.n_red = getattr(args, "n_red", 2)
        self.num_envs = getattr(args, "num_envs", 1)
        self.rollout_group_size = self.n_red * self.num_envs
        self.max_train_steps = args.max_train_steps
        self.max_episode_steps = args.max_episode_steps  # 保存所需
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.save_dir, self.date, self.model_dir = args.save_dir, args.date, args.model_dir

        self.actor = Actor_Gaussian(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a,
                                                eps=1e-5 if self.set_adam_eps else 1e-8)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c,
                                                 eps=1e-5 if self.set_adam_eps else 1e-8)

    def choose_action(self, s):
        s_tensor = torch.tensor(s, dtype=torch.float).to(self.device)
        is_single = len(s_tensor.shape) == 1
        if is_single: s_tensor = s_tensor.unsqueeze(0)

        with torch.no_grad():
            dist = self.actor.get_dist(s_tensor)
            a = dist.sample()
            a = torch.clamp(a, -self.max_action, self.max_action)
            a_logprob = dist.log_prob(a).sum(dim=-1, keepdim=True)

        if is_single:
            return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()
        return a.cpu().numpy(), a_logprob.cpu().numpy()

    def update(
            self,
            replay_buffer,
            total_steps,
            do_lr_decay=True,
            rollout_group_size=None,
            K_epochs_override=None
    ):
        s, a, a_logprob, r, s_next, dw, done = replay_buffer.numpy_to_tensor()
        s = s.to(self.device)
        a = a.to(self.device)
        a_logprob = a_logprob.to(self.device)
        r = r.to(self.device)
        s_next = s_next.to(self.device)
        dw = dw.to(self.device)
        done = done.to(self.device)
        old_group_size = self.rollout_group_size
        # Meta-MAPPO support/query 会拆分并行环境，因此这里允许临时覆盖 GAE 的 rollout 分组大小。
        old_group_size = self.rollout_group_size
        if rollout_group_size is not None:
            self.rollout_group_size = rollout_group_size
        adv, v_target = self.get_adv(s, r, s_next, dw, done)
        self.rollout_group_size = old_group_size
        # Prioritized sampling: emphasize samples with larger absolute advantages.
        adv_abs = torch.abs(adv).squeeze(-1)

        if torch.isnan(adv_abs).any() or adv_abs.sum() <= 1e-8:
            sample_prob = torch.ones_like(adv_abs) / len(adv_abs)
        else:
            priority_prob = adv_abs + 1e-6
            priority_prob = priority_prob / priority_prob.sum()

            uniform_prob = torch.ones_like(priority_prob) / len(priority_prob)
            sample_prob = 0.7 * priority_prob + 0.3 * uniform_prob

        a_loss_sum, c_loss_sum = 0, 0
        K_epochs = self.K_epochs if K_epochs_override is None else K_epochs_override
        buffer_size = s.shape[0]
        mini_batch_size = min(self.mini_batch_size, buffer_size)
        effective_batch_size = min(self.batch_size, buffer_size)
        batch_count = max(1, effective_batch_size // mini_batch_size)

        for _ in range(K_epochs):
            for _ in range(batch_count):
                index = torch.multinomial(sample_prob, mini_batch_size, replacement=False)
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(dim=-1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index]).sum(dim=-1, keepdim=True)
                ratio = torch.exp(a_logprob_now - a_logprob[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.optimizer_actor.zero_grad()
                a_loss.mean().backward()
                if self.use_grad_clip: nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                #c_loss = F.mse_loss(v_target[index], v_s)
                c_loss = F.smooth_l1_loss(v_target[index], v_s)
                self.optimizer_critic.zero_grad()
                c_loss.backward()
                if self.use_grad_clip: nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                a_loss_sum += a_loss.mean().item()
                c_loss_sum += c_loss.item()

        if do_lr_decay and self.use_lr_decay:
            self.lr_decay(total_steps)
        denom = K_epochs * batch_count
        return a_loss_sum / denom, c_loss_sum / denom

    #按智能体/环境分别计算 GAE，避免 red0、red1 以及不同并行环境之间串轨迹。
    def get_adv(self, s, r, s_next, dw, done):
        with torch.no_grad():
            v_s = self.critic(s)
            v_s_next = self.critic(s_next)
            # 【修正】：使用 1.0 - dw，避免 float tensor 按位取反报错
            deltas = r + self.gamma * v_s_next * (1.0 - dw) - v_s
            group_size = self.rollout_group_size
            total_size = deltas.shape[0]

            # 如果长度不能整除 group_size，退回旧写法，避免直接报错
            # 正常情况下，train.py 和 train_parallel.py 应该都能整除
            if total_size % group_size != 0:
                adv = torch.zeros_like(deltas).to(self.device)
                gae = torch.zeros(1, 1, device=self.device)

                for t in reversed(range(total_size)):
                    gae = deltas[t] + self.gamma * self.lamda * gae * (1.0 - done[t])
                    adv[t] = gae

                v_target = adv + v_s
                if self.use_adv_norm:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-5)

                return adv, v_target

            T = total_size // group_size

            # [T, group_size, 1]
            deltas = deltas.view(T, group_size, 1)
            done = done.view(T, group_size, 1)

            adv = torch.zeros_like(deltas).to(self.device)

            # 每个 env-agent 单独维护一个 gae
            gae = torch.zeros(group_size, 1, device=self.device)

            for t in reversed(range(T)):
                gae = deltas[t] + self.gamma * self.lamda * gae * (1.0 - done[t])
                adv[t] = gae

            # 展平成 [batch_size, 1]
            adv = adv.view(-1, 1)
            v_target = adv + v_s

            if self.use_adv_norm: adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        return adv, v_target

    def lr_decay(self, total_steps):
        progress = max(0.0, 1 - total_steps / self.max_train_steps)
        lr_a_now, lr_c_now = self.lr_a * progress, self.lr_c * progress
        for p in self.optimizer_actor.param_groups: p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups: p['lr'] = lr_c_now

    def save(self, agent_id, total_num_steps):
        # 【修正】：动态使用 self.max_episode_steps
        path = f"{self.save_dir}/{self.date}/model/{int(total_num_steps // self.max_episode_steps)}"
        if not os.path.exists(path): os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}/actor_shared.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic_shared.pt")

    def restore(self, agent_id):
        # 【修正】：加入 map_location 以保证跨设备兼容性
        self.actor.load_state_dict(torch.load(f"{self.model_dir}/actor_shared.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{self.model_dir}/critic_shared.pt", map_location=self.device))
