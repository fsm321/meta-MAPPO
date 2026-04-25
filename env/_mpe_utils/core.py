import numpy as np
import math
from numba import njit  # 新增 Numba 导入

@njit(fastmath=True)
def fast_compute_distance_and_angle_scalar(x1, y1, z1, x2, y2, z2, yaw, pitch):
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    fx = math.cos(yaw) * math.cos(pitch)
    fy = math.sin(yaw) * math.cos(pitch)
    fz = math.sin(pitch)

    if dist < 1e-8:
        tx, ty, tz = 0.0, 0.0, 0.0
    else:
        tx, ty, tz = dx / dist, dy / dist, dz / dist

    dot_val = fx * tx + fy * ty + fz * tz
    if dot_val > 1.0:
        dot_val = 1.0
    elif dot_val < -1.0:
        dot_val = -1.0

    return dist, math.acos(dot_val)

class EntityState(object):
    def __init__(self):
        self.p_pos = np.zeros(2)  # 保持 [X, Y] 供渲染器使用(上帝视角)
        self.z_pos = 5.0  # 高度 (Z轴)
        self.p_vel = 0.6  # 标量速度 (V)
        self.yaw = 0.0  # 偏航角 (Heading)
        self.pitch = 0.0  # 俯仰角 (Pitch)

        # 【新增防爆接口】：骗过渲染器，让它在飞机头顶打印空文本
        self.goal = ""
        self.c = ""

class Action(object):
    def __init__(self):
        # 3个维度的控制量: [加速度, 偏航角速度, 俯仰角速度]
        self.u = np.zeros(3)
        # 【防爆接口】: 兼容原版 MPE 的通信维度检查
        self.c = np.zeros(0)


class Entity(object):
    def __init__(self):
        self.name = ''
        self.size = 0.05
        self.color = None
        self.state = EntityState()

        # 【新增防爆接口】：满足 environment.py 解析动作时的检查
        self.accel = None
        self.max_speed = None


class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.movable = True
        self.silent = True
        self.action = Action()
        self.team = 0
        self.done = False

        # ==========================================
        # 【防爆接口】: 满足 environment.py 初始化检查的属性
        # ==========================================
        self.action_callback = None  # 证明此智能体需要被神经网络接管
        self.u_range = 1.0  # 物理动作输出范围归一化为 [-1, 1]
        self.c_range = 1.0  # 通信动作输出范围
        self.u_noise = None  # 物理动作噪声
        self.c_noise = None  # 通信动作噪声
        self.discrete_action = False  # 显式声明我们使用的是连续动作空间


class World(object):
    def __init__(self):
        self.agents = []
        self.dim_c = 0
        self.dim_p = 3
        self.dim_a = 3
        self.dt = 0.1

        # 【新增防爆接口】：骗过渲染器，让它画一个半径为 0 的隐形目标圈
        self.target_radius = 0.0
        self.target_centre = np.zeros(2)

    # ... 下面的 @property 等代码完全保持不变 ...
    # ==========================================
    # 【防爆接口区】：应付 MPE 渲染器和外部包装器的检查
    # ==========================================
    @property
    def entities(self):
        return self.agents

    @property
    def obstacles(self):
        return []  # 告诉渲染器：没有障碍物

    @property
    def landmarks(self):
        return []  # 告诉渲染器：没有目标点

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # ==========================================
    # 物理步进逻辑 (保持不变)
    # ==========================================
    def step(self):
        # Scripted opponents need to refresh their actions every env step.
        for agent in self.scripted_agents:
            if agent.action_callback is None:
                continue
            if getattr(agent, 'is_dead', False):
                agent.action.u = np.zeros(self.dim_a)
            else:
                agent.action.u = np.asarray(agent.action_callback(agent, self), dtype=float)
        for agent in self.agents:
            if agent.movable:
                self.update_agent_state(agent)

    def update_agent_state(self, agent):
        action = agent.action.u
        a = action[0] * 2.0
        w_yaw = action[1] * np.pi/4
        w_pitch = action[2] * np.pi/6
        # ==========================================
        # 限制蓝方（team == 1）的机动性能
        # 让红方的最高速度可达 3.5，蓝方最高被锁死在 2.5
        # ==========================================
        if agent.team == 1:
            agent.state.p_vel = np.clip(agent.state.p_vel + a * self.dt, 0.5, 2.5)
            # 顺便削弱蓝方的转向灵敏度
            agent.state.yaw += w_yaw * self.dt * 0.8
        else:
            agent.state.p_vel = np.clip(agent.state.p_vel + a * self.dt, 0.5, 3.5)
            agent.state.yaw += w_yaw * self.dt

        agent.state.pitch = np.clip(agent.state.pitch + w_pitch * self.dt, -np.pi / 3, np.pi / 3)

        vx = agent.state.p_vel * np.cos(agent.state.yaw) * np.cos(agent.state.pitch)
        vy = agent.state.p_vel * np.sin(agent.state.yaw) * np.cos(agent.state.pitch)
        vz = agent.state.p_vel * np.sin(agent.state.pitch)

        agent.state.p_pos[0] += vx * self.dt
        agent.state.p_pos[1] += vy * self.dt
        agent.state.z_pos += vz * self.dt
