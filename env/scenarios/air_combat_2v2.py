import numpy as np
import math
from env._mpe_utils.core import Agent, World, fast_compute_distance_and_angle_scalar
from env._mpe_utils.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.num_agents, world.collaborative = 4, False
        self.current_task = 0
        world.agents = [Agent() for _ in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name, agent.team = f'uav_{i}', (0 if i < 2 else 1)
            if agent.team == 1: agent.action_callback = self.blue_action_callback
        return world

    def reset_world(self, world, task_id=None):
        self.current_task = task_id if task_id is not None else np.random.randint(0, 3)
        world.blue_tactics_assigned = False
        for agent in world.agents:
            if hasattr(agent, 'last_min_dist'): delattr(agent, 'last_min_dist')
            agent.hp, agent.is_dead, agent.last_action, agent.done = 100.0, False, np.zeros(3), False
            agent.just_killed_by_enemy = False
            if agent.team == 1: agent.combat_mode, agent.tactic = False, None
            agent.color = np.array([0.85, 0.35, 0.35]) if agent.team == 0 else np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np.random.uniform(-5, -2, 2) if agent.team == 0 else np.random.uniform(2, 5, 2)
            agent.state.z_pos, agent.state.p_vel, agent.state.yaw, agent.state.pitch = np.random.uniform(3,
                                                                                                         7), 1.0, np.random.uniform(
                -math.pi, math.pi), 0.0

    def blue_action_callback(self, agent, world):
        action = np.zeros(3)
        red_agents = [a for a in world.agents if a.team == 0 and not a.is_dead]
        if agent.is_dead or not red_agents: return action

        my_pos = (agent.state.p_pos[0], agent.state.p_pos[1], agent.state.z_pos)
        dists = [math.sqrt((r.state.p_pos[0] - my_pos[0]) ** 2 + (r.state.p_pos[1] - my_pos[1]) ** 2 + (
                    r.state.z_pos - my_pos[2]) ** 2) for r in red_agents]
        min_dist, closest_red = min(dists), red_agents[np.argmin(dists)]

        if min_dist > 6.0 and not agent.combat_mode:
            return np.array([1.0, 0.0, 0.0])

        agent.combat_mode = True
        if not getattr(world, 'blue_tactics_assigned', False):
            tactics = [0, 0] if self.current_task == 0 else ([1, 2] if self.current_task == 1 else [0, 1])
            blues = [a for a in world.agents if a.team == 1 and not a.is_dead]
            for i, b in enumerate(blues): b.tactic = tactics[i] if i < len(tactics) else 0
            world.blue_tactics_assigned = True

        rel_p = closest_red.state.p_pos - agent.state.p_pos
        t_yaw = math.atan2(rel_p[1], rel_p[0])
        y_diff = (t_yaw - agent.state.yaw + math.pi) % (2 * math.pi) - math.pi
        z_diff = closest_red.state.z_pos - agent.state.z_pos

        if agent.tactic == 0:
            action[:] = [0.9, (0.8 if y_diff > 0 else -0.8), (0.8 if z_diff > 0 else -0.8)]
        elif agent.tactic == 1:
            action[:] = [0.8, 0.6, (-1.0 if agent.state.z_pos > 5.0 else 1.0)]
        elif agent.tactic == 2:
            f_yaw = t_yaw + math.pi / 2
            fy_diff = (f_yaw - agent.state.yaw + math.pi) % (2 * math.pi) - math.pi
            action[:] = [1.0, (1.0 if fy_diff > 0 else -1.0), 0.0]

        action += np.random.normal(0, 0.15, size=3)
        return np.clip(action, -1.0, 1.0)

    def reward(self, agent, world):
        if agent.is_dead:
            if getattr(agent, 'just_killed_by_enemy', False):
                agent.just_killed_by_enemy = False
                return -20.0
            return 0.0

        if agent.team == 1:
            reds = [r for r in world.agents if r.team == 0 and not r.is_dead]
            if reds:
                d, t_red = min([(math.sqrt(
                    (r.state.p_pos[0] - agent.state.p_pos[0]) ** 2 + (r.state.p_pos[1] - agent.state.p_pos[1]) ** 2 + (
                                r.state.z_pos - agent.state.z_pos) ** 2), r) for r in reds], key=lambda x: x[0])
                _, ata = fast_compute_distance_and_angle_scalar(agent.state.p_pos[0], agent.state.p_pos[1],
                                                                agent.state.z_pos, t_red.state.p_pos[0],
                                                                t_red.state.p_pos[1], t_red.state.z_pos,
                                                                agent.state.yaw, agent.state.pitch)
                if d < 3.5 and ata < math.pi / 6:
                    t_red.hp -= 20.0
                    if t_red.hp <= 0: t_red.is_dead, t_red.done, t_red.just_killed_by_enemy = True, True, True
            return 0.0

        rew = 0.0
        # 1. 软边界惩罚
        if abs(agent.state.p_pos[0]) > 8.0 or abs(
                agent.state.p_pos[1]) > 8.0 or agent.state.z_pos < 1.0 or agent.state.z_pos > 9.0:
            rew -= 1.0

        rew -= 0.05  # 时间惩罚

        # 2. 动作平滑
        if hasattr(agent.action, 'u'):
            rew -= 0.1 * np.sum(np.square(agent.action.u - agent.last_action))
            agent.last_action = np.copy(agent.action.u)

        ens = [e for e in world.agents if e.team == 1 and not e.is_dead]
        if not ens: return rew + 15.0

        d_min, t_en = min([(math.sqrt(
            (e.state.p_pos[0] - agent.state.p_pos[0]) ** 2 + (e.state.p_pos[1] - agent.state.p_pos[1]) ** 2 + (
                        e.state.z_pos - agent.state.z_pos) ** 2), e) for e in ens], key=lambda x: x[0])
        _, ata = fast_compute_distance_and_angle_scalar(agent.state.p_pos[0], agent.state.p_pos[1], agent.state.z_pos,
                                                        t_en.state.p_pos[0], t_en.state.p_pos[1], t_en.state.z_pos,
                                                        agent.state.yaw, agent.state.pitch)

        is_engaging = d_min < 10.0
        am_i_attacking = False

        # 3. 基础生存与高度保持
        if 4.0 < agent.state.z_pos < 6.0 and is_engaging:
            rew += 0.1

        # 4. 势能与攻击奖励
        if d_min < 8.0:
            rew += (math.pi - ata) / math.pi * 2.0
            if ata < math.pi / 6 and d_min < 3.5:
                rew += 5.0
                am_i_attacking = True
                t_en.hp -= 20.0
                if t_en.hp <= 0: t_en.is_dead, t_en.done, rew = True, True, rew + 80.0

        # 5. 多机协同机制 (防撞与集火)
        teammates = [a for a in world.agents if a.team == agent.team and a != agent and not a.is_dead]
        for tm in teammates:
            dist_to_tm = math.sqrt(
                (agent.state.p_pos[0] - tm.state.p_pos[0]) ** 2 + (agent.state.p_pos[1] - tm.state.p_pos[1]) ** 2 + (
                            agent.state.z_pos - tm.state.z_pos) ** 2)
            if dist_to_tm < 0.8:
                rew -= 5.0
            elif 1.5 < dist_to_tm < 4.0 and is_engaging:
                rew += 0.5

            if am_i_attacking:
                _, tm_ata = fast_compute_distance_and_angle_scalar(tm.state.p_pos[0], tm.state.p_pos[1], tm.state.z_pos,
                                                                   t_en.state.p_pos[0], t_en.state.p_pos[1],
                                                                   t_en.state.z_pos, tm.state.yaw, tm.state.pitch)
                dist_to_en = math.sqrt(
                    (t_en.state.p_pos[0] - tm.state.p_pos[0]) ** 2 + (t_en.state.p_pos[1] - tm.state.p_pos[1]) ** 2 + (
                                t_en.state.z_pos - tm.state.z_pos) ** 2)
                if dist_to_en < 6.0 and tm_ata < math.pi / 4:
                    rew += 10.0

        return np.clip(rew, -25.0, 80.0)

    def observation(self, agent, world):
        self_obs = [agent.state.p_pos[0], agent.state.p_pos[1], agent.state.z_pos, agent.state.p_vel, agent.state.yaw,
                    agent.state.pitch]
        other_obs = []
        for other in world.agents:
            if other is agent: continue
            rel = other.state.p_pos - agent.state.p_pos
            rel_z = other.state.z_pos - agent.state.z_pos
            dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel_z ** 2)
            other_obs.extend([rel[0], rel[1], rel_z, dist, (0.0 if other.is_dead else 1.0)])
        return np.concatenate((self_obs, other_obs))

    def done(self, agent, world):
        return True if agent.is_dead else agent.done