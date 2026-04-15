

## 训练

Anaconda终端
```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\7.1
python run_experiments.py
```
## 训练结果

训练过程中的日志会保存在 TensorBoard 中，可以通过以下命令查看：
### 1.奖励曲线，损失曲线，学习率
```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\7.1
tensorboard --logdir=./data/train
```
plot_TensorBoard文件中改数据地址
```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\7.1\result
python plot_TensorBoard.py
```
1_Reward_Curve.png（得分曲线，展示 Meta-MAPPO 更聪明）

2_Actor_Loss_Curve.png（策略损失，展示 Meta-MAPPO 收敛更快）

3_Critic_Loss_Curve.png（价值损失，附赠图，可放论文附录）
### 2.抗噪图、动态恢复图、百局胜率
```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\7.1
python evaluate.py --algo_name MAPPO --model_dir .\data\MAPPO_seed10_0328_015339\model\326100
```
```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\7.1
python evaluate.py --algo_name Meta-MAPPO --model_dir .\data\Meta-MAPPO_seed10_0328_015339\model\327900
```
做完这两步后，目录下就会多出 4 个 .npy 文件：
这些文件必须和 plot_combined_results.py 放在同一个文件夹

更改胜率后运行
```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\7.1\result
python plot_combined.py
```
### 3.3D图
```bash
D:
cd D:\Meta-MAPPO\Meta-MAPPO\7
python plot_3D.py --algo_name Meta-MAPPO --model_dir .\data\Meta-MAPPO_seed10_0320_135440\model\407700
```

## 自定义参数
可以通过命令行参数自定义训练配置：
```bash
python train.py \

    --max_train_steps 500000000 \#最大训练步数
    --max_episode_steps 500 \#每个回合最大步数
    --evaluate_freq 500 \#评估频率
    --save_freq  1000 \#模型保存频率
    --buffer_size 4000 \#经验回放池容量
     --mini_batch_size 256 \#迷你批次大小
    --hidden_width 128 \#隐藏层神经元数量
    --lr_a  1e-4 \#actor学习率
    --lr_c  3e-4 \#critic学习率
    --epsilon 0.1 \#裁剪比例
    --K_epochs 5 \#同一批数据会过 5 遍网络
    
```

### 主要参数说明

- `--scenario_name`: 场景名称（默认: `simple_spread`）
- `--max_train_steps`: 最大训练步数（默认: 51000000）
- `--max_episode_steps`: 每个回合最大步数（默认: 125）
- `--policy_dist`: 策略分布类型，`Gaussian` 或 `Beta`（默认: `Gaussian`）
- `--restore`: 是否加载已有模型（默认: `False`）
- `--save_dir`: 模型保存目录（默认: `./data`）
- `--model_dir`: 模型加载目录

## 项目结构

```
Meta-MAPPO/
├── algorithms/                      # 算法核心模块：存放各类强化学习与元学习策略
│   ├── mappo.py                     # 基线算法：多智能体近端策略优化 (MAPPO)
│   └── meta_mappo.py                # 核心创新：基于元学习的MAPPO (Meta-MAPPO)
│
├── env/                             # 仿真环境模块：定义智能体与环境的交互逻辑
│   ├── MPE_env.py                   # 多智能体粒子环境 (MPE) 接口的二次封装
│   ├── environment.py               # 强化学习标准交互环境包装器 (Wrapper)
│   ├── scenarios/                   # 具体任务场景定义
│   │   └── air_combat_2v2.py        # 核心场景：2v2 多无人机协同空战定制化环境
│   └── _mpe_utils/                  # MPE环境底层依赖工具
│       ├── core.py                  # 物理引擎与实体状态核心逻辑
│       ├── rendering.py             # 画面渲染模块
│       ├── scenario.py              # 场景构建基类
│       └── secrcode.ttf             # 渲染所需的字体文件
│
├── result/                          # 实验结果分析与可视化模块
│   ├── plot_combined.py             # MAPPO vs Meta-MAPPO对比图表绘制
│   └── plot_TensorBoard.py          # TensorBoard 训练奖励曲线的平滑与可视化提取
│
├── utils/                           # 通用基础工具包
│   ├── normalization.py             # 数据归一化工具 (状态/奖励归一化，对稳定性至关重要)
│   └── replaybuffer.py              # 经验回放池 (用于轨迹采样与元学习任务的数据收集)
│
├── train.py                         # 模型主训练脚本 (包含环境初始化、模型实例化及元训练主循环)
├── evaluate.py                      # 模型评估脚本 (测试Meta-MAPPO在陌生空战场景中的适应能力)
├── run_experiments.py               # 自动化批量实验脚本 (用于跑多组随机种子或调参)
├── plot_3D.py                       # 3D轨迹或状态空间可视化工具
├── README.md                        # 项目说明文档 (建议包含环境配置、运行指令与算法简介)
```

## 10 种训练技巧

1. **Advantage Normalization** - 优势函数归一化
2. **State Normalization** - 状态归一化
3. **Reward Normalization** - 奖励归一化
4. **Reward Scaling** - 奖励缩放
5. **Policy Entropy** - 策略熵正则化
6. **Learning Rate Decay** - 学习率衰减
7. **Gradient Clip** - 梯度裁剪
8. **Orthogonal Initialization** - 正交初始化
9. **Adam Optimizer Epsilon Parameter** - Adam 优化器参数设置
10. **Tanh Activation Function** - Tanh 激活函数
