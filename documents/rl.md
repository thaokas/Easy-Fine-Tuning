# 强化学习关键方法综述

> 本文系统整理了强化学习（Reinforcement Learning, RL）的关键方法，涵盖从经典算法到 2025–2026 年最新进展，包括价值方法、策略方法、Actor-Critic、基于模型的方法、离线RL、以及面向大语言模型的对齐与推理训练方法。

---

## 目录

1. [基础概念与分类](#1-基础概念与分类)
2. [基于价值的方法](#2-基于价值的方法)
3. [基于策略的方法](#3-基于策略的方法)
4. [Actor-Critic 方法](#4-actor-critic-方法)
5. [基于模型的方法](#5-基于模型的方法)
6. [离线/批量强化学习](#6-离线批量强化学习)
7. [面向 LLM 的对齐与推理 RL](#7-面向-llm-的对齐与推理-rl)
8. [其他前沿方向](#8-其他前沿方向)
9. [算法选择指南](#9-算法选择指南)
10. [参考文献](#10-参考文献)

---

## 1. 基础概念与分类

### 核心要素

- **智能体（Agent）**：学习和决策的主体
- **环境（Environment）**：智能体所处的外部世界
- **状态（State）**：环境在某一时刻的描述，记作 $s_t \in \mathcal{S}$
- **动作（Action）**：智能体在状态下采取的行为，记作 $a_t \in \mathcal{A}$
- **奖励（Reward）**：环境对动作的即时反馈，记作 $r_t \in \mathbb{R}$
- **策略（Policy）**：状态到动作的映射，$\pi(a|s)$（随机策略）或 $a = \mu(s)$（确定性策略）
- **价值函数（Value Function）**：从状态（或状态-动作对）出发的期望累积回报
  - 状态价值函数：$V^\pi(s) = \mathbb{E}_\pi[\sum_{k=0}^\infty \gamma^k r_{t+k} \mid s_t = s]$
  - 动作价值函数：$Q^\pi(s,a) = \mathbb{E}_\pi[\sum_{k=0}^\infty \gamma^k r_{t+k} \mid s_t = s, a_t = a]$

### 算法分类总览

```
强化学习
├── 无模型（Model-Free）
│   ├── 基于价值（Value-Based）── Q-Learning, DQN, Rainbow
│   ├── 基于策略（Policy-Based）── REINFORCE, TRPO, PPO
│   └── Actor-Critic ── A2C/A3C, DDPG, TD3, SAC
├── 基于模型（Model-Based）
│   ├── 学习模型 + 规划 ── Dyna, MuZero
│   └── 世界模型 + 想象 ── DreamerV3
└── 离线/Batch RL ── CQL, IQL, Decision Transformer
```

---

## 2. 基于价值的方法

> 核心思想：学习最优动作价值函数 $Q^*(s,a)$，策略由 $\arg\max_a Q(s,a)$ 隐式定义。

### 2.1 Q-Learning（表格方法）

最基础的离线（Off-Policy）价值方法，使用贝尔曼最优方程直接学习 $Q^*$：

**更新规则：**

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$$

**关键特性：**
- **Off-Policy**：学习最优策略独立于行为策略
- **Model-Free**：不需要环境动态模型
- **收敛性**：在表格设定下保证收敛，但无法扩展到高维状态空间

**局限：** $|\mathcal{S}| \times |\mathcal{A}|$ 表格在高维问题中不可行。

---

### 2.2 Deep Q-Network (DQN)

**提出者：** Mnih et al. (2013/2015, DeepMind)

使用神经网络 $Q_\theta(s,a)$ 近似动作价值函数，在 Atari 游戏上取得突破性成果。

**三大核心创新：**

| 创新 | 作用 |
|------|------|
| **经验回放（Experience Replay）** | 存储转移 $(s,a,r,s',done)$ 到回放缓冲区，随机采样训练，打破数据时序相关性 |
| **目标网络（Target Network）** | 维护独立的目标网络 $Q_{\theta^-}$，周期性从在线网络更新，稳定训练目标 |
| **损失函数** | $\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}[(r + \gamma\max_{a'}Q_{\theta^-}(s',a') - Q_\theta(s,a))^2]$ |

**已知问题：**
- **Q 值高估（Overestimation Bias）**：$\max$ 算子放大正向估计误差
- **均匀采样低效**：所有转移被同等对待
- **表示能力受限**：同一网络需同时建模状态价值和动作优势

---

### 2.3 DQN 改进系列

#### Double DQN
分离动作选择与评估，消除高估偏差：

$$a^* = \arg\max_{a'} Q_\theta(s', a') \quad \text{（在线网络选择）}$$

$$y = r + \gamma \cdot Q_{\theta^-}(s', a^*) \quad \text{（目标网络评估）}$$

#### Dueling DQN
将 Q 函数分解为状态价值 $V(s)$ 和动作优势 $A(s,a)$ 两个流：

$$Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')\right)$$

优势：在动作选择无关紧要的状态下（如 Pong 中球远离球拍时），网络不需要浪费容量区分近似等效的动作。

#### Prioritized Experience Replay (PER)
根据 TD 误差的大小为转移分配优先级：

$$p_i = |\delta_i| + \varepsilon, \quad P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

使用 Sum-Tree 数据结构实现 $O(\log N)$ 的采样和更新。需配合重要性采样（Importance Sampling）修正优先级引入的偏差。

#### Multi-Step Learning
使用 $n$ 步回报替代单步 TD 目标，平衡偏差-方差权衡：

$$y = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q_{\theta^-}(s_{t+n}, a')$$

#### Distributional RL (C51)
预测回报的**完整概率分布**而非仅期望值。使用 $N=51$ 个固定原子（取值区间）上的分类分布表示 $Z(s,a)$，损失函数为交叉熵。

#### Noisy Nets
用可学习的参数化噪声替代 $\varepsilon$-greedy 探索：

$$W = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,1)$$

探索变为**状态条件化**（在不熟悉状态中自动探索更多），无需手动退火。

---

### 2.4 Rainbow DQN

**提出者：** Hessel et al. (2018)

将上述六种改进整合到一个统一架构中，在 Atari 2600 基准上以更少样本取得更高最终奖励。

| 组件 | 解决问题 | 消融实验中移除的影响 |
|------|---------|-------------------|
| **Prioritized Replay** | 采样效率低 | **最大** |
| **Multi-step Learning** | TD 目标偏差 | **很大** |
| **Distributional RL** | 忽略回报分布 | 中等 |
| **Noisy Nets** | 盲目探索 | 有帮助，少数游戏有轻微负面 |
| **Dueling** | 低效 Q 值表示 | 部分游戏显著 |
| **Double DQN** | Q 值高估 | 在有 Distributional RL 时影响较小 |

> **总结：** Rainbow 是离散动作空间中的首选基线方法，组件可模块化选择以适应具体场景。

---

## 3. 基于策略的方法

> 核心思想：直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升优化期望回报。

### 3.1 REINFORCE（蒙特卡洛策略梯度）

最基础的策略梯度方法：

$$\nabla J(\theta) = \mathbb{E}_\pi\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

其中 $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ 为蒙特卡洛回报。

**优缺点：** 简单直接，无偏估计；但方差极大，收敛缓慢。

---

### 3.2 TRPO（Trust Region Policy Optimization）

**提出者：** Schulman et al. (2015)

通过 KL 散度约束限制策略更新幅度，保证单调改进：

$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}(s,a)\right] \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) \leq \delta$$

使用二阶优化（共轭梯度 + 线搜索），稳定但计算开销大。

---

### 3.3 PPO（Proximal Policy Optimization）

**提出者：** Schulman et al. (2017)

用**截断代理目标**替代 TRPO 的 KL 约束，只需一阶优化：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为概率比率，$\varepsilon$ 通常取 0.1–0.2。

**PPO 成为"黄金标准"的原因：**
- 实现简单，仅需一阶优化
- 稳定性远超 REINFORCE，接近 TRPO
- 结合 GAE（Generalized Advantage Estimation）进行优势估计
- 被广泛用于 RLHF（ChatGPT 的对齐训练）

**2025 年 PPO 变体：**

| 变体 | 改进点 |
|------|--------|
| **TRGPPO** | 自适应、动作特定的信任域，替代固定裁剪 |
| **SPO** | 二次惩罚替代比率裁剪，解决零梯度问题，在 Atari/MuJoCo 上优于 PPO |
| **ExploRLer** | 可插拔的迭代级探索管线，探测梯度更新附近的未探索区域 |

---

## 4. Actor-Critic 方法

> 核心思想：结合策略方法（Actor）和价值方法（Critic），Actor 输出动作，Critic 评估动作好坏。

### 4.1 A2C / A3C

- **A2C（Advantage Actor-Critic）**：同步版本，使用优势函数 $A(s,a) = Q(s,a) - V(s)$ 减少策略梯度方差
- **A3C（Asynchronous）**：多个 worker 并行在环境副本中采样，异步更新全局网络，利用并行性加速训练

---

### 4.2 DDPG（Deep Deterministic Policy Gradient）

**提出者：** Lillicrap et al. (2016)

将 DQN 扩展到连续动作空间，使用确定性策略 $\mu_\theta(s)$：

- **Actor**：确定性策略网络，输出具体动作值
- **Critic**：Q 网络，评估动作价值
- 采用 DQN 的经验回放和目标网络技术

**局限：** 存在严重的 Q 值高估问题，训练不稳定。

---

### 4.3 TD3（Twin Delayed DDPG）

**提出者：** Fujimoto et al. (2018)

对 DDPG 的三项关键改进：

| 改进 | 细节 |
|------|------|
| **双 Critic（Clipped Double Q-Learning）** | 两个独立 Q 网络，取最小值作为目标，减少高估 |
| **延迟策略更新** | Actor 更新频率低于 Critic（通常每 2 次 Critic 更新 1 次） |
| **目标策略平滑** | 向目标动作添加小噪声，平滑 Q 值估计 |

**2025 年地位：** 适合需要稳定确定性控制的场景（如机器人抓取预定位）。

---

### 4.4 SAC（Soft Actor-Critic）

**提出者：** Haarnoja et al. (2018)

**核心创新——最大熵强化学习：** 在奖励之外最大化策略熵，鼓励探索：

$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\pi}\left[r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

**关键特性：**
- **随机策略**：输出高斯分布，使用重参数化技巧保证梯度流通
- **Soft Bellman 备份**：$Q(s,a) = r + \gamma[\min(Q_1', Q_2') - \alpha\log\pi(a'|s')]$
- **自动温度调节**：熵系数 $\alpha$ 自动学习以维持目标熵水平

**2025 年地位：** 连续控制任务的**事实标准**（SOTA），在 Humanoid、LunarLander 等复杂基准上平均回报最高。

**2025 年改进：**
- **DSAC-T / DSACv2**：学习连续高斯值分布替代点估计，在所有环境上匹配或超越 SAC、TD3、DDPG、TRPO、PPO

---

### 连续控制算法对比

| 算法 | 策略类型 | 核心创新 | 探索能力 | 稳定性 | 适用场景 |
|------|---------|---------|---------|--------|---------|
| **DDPG** | 确定性 | DQN + 连续动作 | 低 | 低 | 简单基线 |
| **TD3** | 确定性 | 双Critic + 延迟更新 + 平滑 | 中 | 高 | 稳定确定性控制 |
| **SAC** | 随机（高斯） | 最大熵 + 自动温度调节 | **高** | 高 | 复杂连续控制 |

---

## 5. 基于模型的方法

> 核心思想：学习环境动态模型，在模型中进行规划或训练，极大提升样本效率。

### 5.1 Dyna 架构

经典框架，在学习过程中交替使用真实经验和模拟经验（经由学习模型生成）进行规划。

---

### 5.2 MuZero

**提出者：** Schrittwieser et al. (2019, DeepMind, *Nature*)

学习环境的**隐式模型**——仅预测对规划有用的信息（奖励、价值、策略），而不重建观测：

- **Representation**：将原始观测 $o_t$ 编码为隐状态 $s_t$
- **Dynamics**：预测下一隐状态 $s_{t+1}$ 和即时奖励 $r_t$
- **Prediction**：预测策略 $p_t$ 和价值 $v_t$
- **Planning**：使用蒙特卡洛树搜索（MCTS）在隐空间中进行决策时规划

**成就：** 统一框架掌握围棋、国际象棋、将棋和 Atari，无需知道游戏规则。

---

### 5.3 DreamerV3

**提出者：** Hafner et al. (2025, DeepMind, *Nature*)

**核心思想：** 从像素输入学习紧凑的世界模型，然后在模型的"想象轨迹"中训练策略——无需与真实环境交互。

**架构（3 个并行训练组件）：**

| 组件 | 功能 |
|------|------|
| **世界模型（RSSM）** | 编码器压缩观测 → 循环状态空间模型预测未来隐状态 + 奖励 + 继续标志 → 解码器重建观测 |
| **Actor** | 在世界模型生成的想象轨迹上训练的策略网络 |
| **Critic** | 在想象轨迹上训练的价值估计器，使用 $\lambda$-回报 |

**跨域稳定性的关键创新：**
- **Symlog 变换**：对称压缩标量值以处理跨任务的不同量级
- **KL 平衡 + Free Bits**：防止隐表示坍缩
- **Two-Hot 值回归**：将值预测视为分类问题以增强鲁棒性
- **Unimix**：向分类输出混合 1% 均匀噪声

**关键成果：**
- **150+ 任务**，8 个基准，单组超参数超越特定领域专家
- Atari 上以更少计算量超过 MuZero
- **首个**在 Minecraft 中从零采集钻石的算法（1 GPU，约 10 小时），无需人类数据或课程学习
- 模型规模 1M–400M 参数，越大越好

### DreamerV3 vs MuZero

| 维度 | DreamerV3 | MuZero |
|------|-----------|--------|
| **世界模型类型** | 生成式（重建观测） | 预测式（仅奖励+价值+策略） |
| **规划方法** | 想象展开（可微分） | MCTS（离散搜索） |
| **动作空间** | 连续 + 离散 | 主要离散 |
| **样本效率** | **最高** | 较高 |
| **推理成本** | 低（单次前向传播） | 高（每步 MCTS） |
| **领域通用性** | 150+ 任务，单套配置 | 强但较窄 |
| **Minecraft 钻石** | ✅ | ❌ |

---

## 6. 离线/批量强化学习

> 核心思想：从固定静态数据集中学习策略，不与环境交互。关键挑战是分布外动作的 Q 值高估。

### 6.1 CQL（Conservative Q-Learning）

**提出者：** Kumar et al. (2020)

对分布外动作的 Q 值施加惩罚，防止高估：

$$\mathcal{L}_{CQL} = \mathcal{L}_{Q} + \alpha \left(\mathbb{E}_{a\sim\mu} [Q(s,a)] - \mathbb{E}_{a\sim\mathcal{D}} [Q(s,a)]\right)$$

**2025 年地位：** 平衡性最强的离线 RL 算法，跨数据质量表现稳健。

---

### 6.2 IQL（Implicit Q-Learning）

**提出者：** Kostrikov et al. (2021)

无需查询分布外动作即可学习 Q 函数，使用期望分位数回归：

- 在**稠密奖励 + 高质量数据**场景中表现最佳
- 计算效率最高（训练时间约 CQL 的 40%）

---

### 6.3 Decision Transformer

**提出者：** Chen et al. (2021)

将 RL 重新定义为**序列建模问题**，使用 Transformer 架构自回归预测动作：

$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots)$$

其中 $\hat{R}_t$ 为期望回报（Return-to-Go）。

**2025 年研究发现：**
- 对**稀疏奖励**场景低敏感性，方差不高的表现
- 但计算开销显著更大（训练时间约 CQL 的 1.5 倍）
- 有工作质疑：简单 MLP + 过滤行为克隆（FBC）在稀疏和稠密奖励场景中均可达到竞争力或更优性能

### 离线 RL 算法对比

| 算法 | 核心思想 | 最佳场景 | 计算开销 |
|------|---------|---------|---------|
| **CQL** | 惩罚 OOD 动作 Q 值 | 各数据质量下均衡 | 中等 |
| **IQL** | 隐式 Q 学习，不查询未见动作 | 稠密奖励 + 高质量数据 | **最低** |
| **Decision Transformer** | RL as 序列建模 | 稀疏奖励 + 混合质量数据 | 最高 |
| **FBC** | 过滤低回报轨迹 + 行为克隆 | 通用（简单有效） | 低 |

---

## 7. 面向 LLM 的对齐与推理 RL

> 2024–2026 年最重要的 RL 应用方向。使用 RL 使大语言模型与人类偏好对齐，并通过可验证奖励激发推理能力。

### 演进路线

```
SFT（模仿学习）→ RLHF/PPO（对齐）→ DPO（简化对齐）→ GRPO/DAPO（推理涌现）→ RLVR（2026 前沿）
```

---

### 7.1 RLHF（Reinforcement Learning from Human Feedback）

**经典三阶段管线（InstructGPT / ChatGPT）：**

1. **SFT（Supervised Fine-Tuning）**：在高质量人类示范上微调
2. **奖励模型训练**：从成对人类偏好比较中学习 Bradley-Terry 奖励模型
3. **PPO 优化**：最大化奖励模型得分，同时通过 KL 惩罚保持接近 SFT 模型

| 维度 | 详情 |
|------|------|
| 所需模型数 | **4**（策略、参考、奖励、价值/Critic） |
| 计算成本 | 极高（~4x 单模型内存） |
| 训练稳定性 | 低——PPO 调参困难 |
| 人类标注 | 需要偏好对 |
| 探索能力 | ✅ 在线 RL 可探索训练数据之外 |

**瓶颈：** 4 模型内存、奖励黑客、分布偏移。

---

### 7.2 DPO（Direct Preference Optimization）

**提出者：** Rafailov et al. (2023)

**核心洞察：** RLHF 目标存在闭式最优解，可直接在偏好数据上优化策略，无需独立的奖励模型或 RL 循环。

**损失函数：**

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

| 维度 | 详情 |
|------|------|
| 所需模型数 | **2**（策略 + 冻结参考） |
| 计算成本 | 中等（~2x 内存，无需 rollout） |
| 训练稳定性 | **高**——监督式损失 |
| 人类标注 | 需要偏好对 |
| 探索能力 | ❌ 离线——受数据集质量限制 |

**优势：** RLHF 收益的 90%，仅需 10% 的成本。Zephyr 模型是其旗舰成功案例。

**DPO 变体：**
- **SimPO**：移除参考模型依赖，使用长度归一化奖励
- **ORPO**：单阶段组合 SFT 和偏好优化
- **IPO**：基于 MSE 的公式，更好的正则化
- **KTO**：使用二元（好/坏）标签替代成对比较，基于前景理论

---

### 7.3 GRPO（Group Relative Policy Optimization）

**提出者：** DeepSeek (2024)，用于训练 DeepSeek-R1

**核心创新：** 彻底移除 Critic 网络，使用组内相对优势估计：

1. 对每个提示采样 $G$ 个回答（通常 8–64 个）
2. 用**可验证奖励函数**（而非学习到的奖励模型）打分
3. 通过组归一化计算优势：$A_i = \frac{r_i - \text{mean}(r_1\ldots r_G)}{\text{std}(r_1\ldots r_G)}$
4. 应用 PPO 式截断目标——无需 Critic 模型

| 维度 | 详情 |
|------|------|
| 所需模型数 | **1–2**（策略 + 可选参考；**无 Critic**） |
| 计算成本 | 中等（PPO 的一半——无 Critic） |
| 人类标注 | ❌ 不需要——基于规则的可验证奖励 |
| 探索能力 | ✅ 在线 RL，多次 rollout |
| 最佳场景 | 数学、代码、逻辑、推理任务 |

**革命性成果：**
- DeepSeek-R1 证明：纯 RL + 可验证奖励可以**涌现推理能力**（自我反思、策略适应），无需任何人类标注推理轨迹
- AIME 得分：15.6% → 71% → 86.7%（多数投票）

---

### 7.4 DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）

**提出者：** 字节跳动/清华 (2025)

通过 4 项技术稳定长链推理 RL 训练：

| 技术 | 作用 |
|------|------|
| **Clip-Higher** | 防止熵坍缩 |
| **Dynamic Sampling** | 过滤无信息量样本 |
| **Token-level Policy Gradient** | 对长 CoT 序列至关重要 |
| **Overlong Reward Shaping** | 处理截断回答 |

**成果：** Qwen2.5-32B → AIME 50 分，训练步数比 DeepSeek-R1-Zero 少 **50%**。

---

### 7.5 RLVR（Reinforcement Learning with Verifiable Rewards）

2025–2026 年主导范式。核心洞察：对于数学、代码、推理——自动验证器（单元测试、证明检查器、数学评估器）提供比人类更快、更便宜、更一致的奖励信号。

---

### LLM 对齐方法全对比

| 维度 | RLHF (PPO) | DPO | GRPO | DAPO |
|------|-----------|-----|------|------|
| **奖励来源** | 学习到的 RM（人类偏好） | 偏好对中隐式 | 可验证规则/函数 | 可验证规则/函数 |
| **需要人类标注？** | ✅ | ✅ | ❌ | ❌ |
| **Critic 模型？** | ✅ | ❌ | ❌ | ❌ |
| **奖励模型？** | ✅ | ❌ | ❌ | ❌ |
| **内存中模型数** | 4 | 2 | 1–2 | 1–2 |
| **在线探索？** | ✅ | ❌ | ✅ | ✅ |
| **训练稳定性** | 低 | 高 | 中高 | 中高 |
| **推理能力提升** | 有限 | 有限 | **显著** | **显著** |
| **通用对齐** | 优秀 | 优秀 | 差（需可验证信号） | 差 |
| **代表模型** | InstructGPT, Llama 2-Chat | Llama 3, 开源模型 | DeepSeek-R1 | Qwen2.5 + DAPO |

### 2026 生产级后训练管线

```
SFT（1-10M 样本，格式学习）
  → 偏好优化（DPO/SimPO/KTO，对齐风格与安全）
  → 强化学习（GRPO/DAPO + 可验证奖励，激发推理）
```

---

## 8. 其他前沿方向

### 8.1 多智能体强化学习（MARL）

- **层级 MARL**：高层规划（常由 LLM 负责）+ 低层 RL 执行
- **联赛训练（League Training）**：AlphaStar 用于星际争霸 II，智能体在多样化群体中对抗训练
- 基于注意力机制的通信机制，实现可扩展多智能体协调
- 2025 年趋势：LLM 智能体 + RL 的混合架构

### 8.2 安全强化学习（Safe RL）

工业部署的关键要求：
- **拉格朗日方法**：将安全约束转换为可微分惩罚项
- **障碍函数（Barrier Function）**：在策略输出层直接添加安全边界
- 符合 ISO 26262 等自动驾驶安全标准

### 8.3 元强化学习（Meta-RL）

快速适应新任务，只需极少数据：
- 通过大模型进行上下文编码
- 基于梯度的少样本微调适应
- 2025 年：新环境所需样本减少高达 80%

### 8.4 分布式强化学习

- **架构**：环境（并行）→ 采样器 → 回放缓冲区 → 学习器 → 目标更新 → 评估
- 常用框架：RLlib (Ray)、SeedRL
- 关键收益：解耦采样和训练，GPU 利用率大幅提升

### 8.5 量子强化学习（新兴）

- 使用 RL（PPO、A2C）进行参数化量子态准备和电路编译
- 已演示在超导噪声下可靠的 Bell 态重建

---

## 9. 算法选择指南

### 按场景选择

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 连续控制 / 机器人 | **SAC** | 最佳样本效率，自动熵调节 |
| 离散动作空间 / 游戏 | **Rainbow DQN** | SOTA 性能，模块化组件 |
| 大模型对齐（通用） | **PPO (RLHF)** 或 **DPO** | 稳定性 vs. 简单性的权衡 |
| 推理模型训练（数学/代码） | **GRPO** 或 **DAPO** | 无需 Critic，可验证奖励 |
| 样本受限环境 | **DreamerV3** 或 **CQL** | 基于模型 / 离线学习 |
| 多机器人协调 | **层级 MARL** | 处理组合复杂度 |
| 安全关键应用 | **Safe RL（拉格朗日）** | 显式约束满足 |
| 快速适应新任务 | **Meta-RL** | 少样本策略迁移 |
| 基础指令遵循 | **SFT + LoRA** | 最简单、成本最低 |
| 安全性 / 风格对齐 | **DPO** | 稳定、无需 RM、LoRA 可用 |
| 长链推理 | **DAPO** | 稳定长 CoT 训练 |

### 按目标选择 LLM 对齐方法

| 目标 | 推荐方法 |
|------|---------|
| 基础指令遵循 | SFT（配合 LoRA） |
| 安全 / 风格 / 语调对齐 | **DPO** |
| 最优对齐质量 | **RLHF (PPO)** |
| 数学 / 代码 / 逻辑推理 | **GRPO** |
| 通用对齐 + 推理 | **SFT → DPO → GRPO**（三段组合） |

---

## 10. 参考文献

### 综述与教材
- *"Reinforcement Learning: An Introduction"* (2nd Ed.) — Sutton & Barto (2018)
- *Mastering Diverse Control Tasks through World Models* — Hafner et al., *Nature* (2025)
- *A Comprehensive Review of Reinforcement Learning* — arXiv:2510.21758 (2025)
- *A Technical Survey of Reinforcement Learning Techniques for Large Language Models* — arXiv:2507.04136 (2025)

### 经典算法
- *Playing Atari with Deep Reinforcement Learning* — Mnih et al. (2013)
- *Human-level control through deep reinforcement learning* — Mnih et al., *Nature* (2015)
- *Rainbow: Combining Improvements in Deep Reinforcement Learning* — Hessel et al. (2018)
- *Trust Region Policy Optimization* — Schulman et al. (2015)
- *Proximal Policy Optimization Algorithms* — Schulman et al. (2017)
- *Soft Actor-Critic* — Haarnoja et al. (2018)
- *Addressing Function Approximation Error in Actor-Critic Methods (TD3)* — Fujimoto et al. (2018)

### 基于模型
- *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)* — Schrittwieser et al., *Nature* (2020)
- *Mastering Diverse Control Tasks through World Models (DreamerV3)* — Hafner et al., *Nature* (2025)

### 离线 RL
- *Conservative Q-Learning for Offline Reinforcement Learning* — Kumar et al. (2020)
- *Offline Reinforcement Learning as One Big Sequence Modeling Problem (Decision Transformer)* — Chen et al. (2021)
- *A Comparison Between Decision Transformers and Traditional Offline RL Algorithms* — arXiv:2511.16475 (2025)

### LLM 对齐与推理 RL
- *Training language models to follow instructions with human feedback (InstructGPT)* — Ouyang et al. (2022)
- *Direct Preference Optimization* — Rafailov et al. (2023)
- *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — DeepSeek (2025)
- *DAPO: Dynamic Sampling and Clipping for LLM Reasoning RL* — ByteDance/Tsinghua (2025)
- *Post-Training in 2026: GRPO, DAPO, RLVR & Beyond* — [llm-stats.com](https://llm-stats.com) (2026)
