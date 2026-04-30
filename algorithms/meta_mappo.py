import torch
import torch.nn as nn
from algorithms.mappo import MAPPO_Continuous


class Meta_MAPPO_Continuous(MAPPO_Continuous):
    def __init__(self, args):
        super(Meta_MAPPO_Continuous, self).__init__(args)
        self.initial_entropy = args.entropy_coef

    def get_weights(self):
        # ==========================================
        # 【核心修正】：移除 .cpu()，直接在显存中 detach 和 clone！
        # 避免每次 meta_update 都在 GPU 和 CPU 之间来回搬运几十MB的参数，极大提升速度
        # ==========================================
        actor_weights = {k: v.detach().clone() for k, v in self.actor.state_dict().items()}
        critic_weights = {k: v.detach().clone() for k, v in self.critic.state_dict().items()}
        return actor_weights, critic_weights

    def meta_update(self, old_weights, meta_lr):
        old_actor, old_critic = old_weights
        with torch.no_grad():
            # Actor 一阶元更新 (基于 Reptile 的一阶 MAML 思想)
            for name, param in self.actor.named_parameters():
                if name in old_actor:
                    param.data = old_actor[name] + meta_lr * (param.data - old_actor[name])

            # Critic 一阶元更新
            for name, param in self.critic.named_parameters():
                if name in old_critic:
                    param.data = old_critic[name] + meta_lr * (param.data - old_critic[name])

    def meta_train_step(
            self,
            support_buffer,
            query_buffer,
            total_steps,
            meta_lr,
            support_group_size,
            query_group_size,
            inner_epochs=1,
            outer_epochs=1
    ):
        """
        Support/Query 一阶元更新：
        1. 保存旧参数
        2. support buffer 上做内层适应
        3. query buffer 上做外层验证更新
        4. 旧参数向 query 更新后的参数移动
        """
        old_weights = self.get_weights()

        # 先在 support 轨迹上做快速适应，再在 query 轨迹上检验适应后的策略。
        support_actor_loss, support_critic_loss = self.update(
            support_buffer,
            total_steps,
            do_lr_decay=False,
            rollout_group_size=support_group_size,
            K_epochs_override=inner_epochs
        )

        query_actor_loss, query_critic_loss = self.update(
            query_buffer,
            total_steps,
            do_lr_decay=False,
            rollout_group_size=query_group_size,
            K_epochs_override=outer_epochs
        )

        self.meta_update(old_weights, meta_lr)

        # 整个 meta step 只衰减一次学习率，避免 support/query 双重衰减。
        if self.use_lr_decay:
            self.lr_decay(total_steps)

        return (
            support_actor_loss,
            support_critic_loss,
            query_actor_loss,
            query_critic_loss
        )

    def lr_decay(self, total_steps):
        # 学习率与探索率（熵）的线性衰减
        progress = max(0.0, 1 - total_steps / self.max_train_steps)
        lr_a_now, lr_c_now = self.lr_a * progress, self.lr_c * progress

        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

        # 保证最小有 0.001 的探索底线，防止策略过早固化
        self.entropy_coef = max(0.001, self.initial_entropy * progress)
