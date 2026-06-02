# Actor-Critic、A2C 与 A3C 学习笔记

> 适用目标：建立理解 PPO 与大模型 RL 后训练所需的强化学习基础。  
> 资料核验日期：2026-05-24

## 1. 先澄清名称

课程中的 “A-C 方法” 通常指 **Actor-Critic（演员-评论家）方法**，它是一类架构或算法思想；A3C 与 A2C 是基于优势函数的具体实现路线：

| 名称 | 全称 | 核心特点 |
| --- | --- | --- |
| Actor-Critic | Actor-Critic | policy 与 value 联合学习的框架 |
| A3C | Asynchronous Advantage Actor-Critic | 多 actor 异步更新共享参数 |
| A2C | Advantage Actor-Critic | 同步收集多 actor 轨迹后统一更新 |

## 2. 为什么要有 Actor 和 Critic

在策略梯度方法中，Actor 参数化策略 `pi_theta(a|s)`，选择动作。最基础的更新会用一次轨迹获得的回报 `G_t` 加权 log probability：

$$
\nabla_\theta J(\theta)
\approx
\nabla_\theta \log \pi_\theta(a_t|s_t)G_t
$$

问题是 `G_t` 的随机性很强，梯度方差大。Critic 估计状态价值：

$$
V_\phi(s_t) \approx \mathbb{E}[G_t \mid s_t]
$$

Actor 不再只问“最终拿了多少分”，而是问“这个动作比当前状态下的通常表现好多少”：

$$
A(s_t,a_t) = Q(s_t,a_t) - V(s_t)
$$

实际中常以 TD error 作为一步优势估计：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

`delta_t > 0` 表示动作结果超出 Critic 预期，应提高其概率；反之降低。

## 3. Actor-Critic 的更新结构

```text
状态 s_t -> Actor: pi(a|s) -> 动作 a_t -> 环境 -> 奖励 r_t, 新状态 s_(t+1)
      \-> Critic: V(s_t) -----------------> 估计 advantage
```

典型联合损失包含：

- policy loss：以 advantage 引导 Actor；
- value loss：使 Critic 的 `V(s)` 接近目标 return；
- entropy bonus：防止策略太早退化为过于确定、失去探索。

## 4. A3C：异步的优势 Actor-Critic

Mnih et al. 在 ICML 2016 论文中提出异步深度强化学习框架，并报告 asynchronous actor-critic 是其中表现最好的变体：

```text
多个 worker 各自在环境中探索
    -> 每个 worker 计算本地梯度
    -> 异步写入共享全局模型
    -> 周期性从全局模型同步参数
```

### 关键价值

- 不依赖 experience replay，通过多个并行环境降低轨迹相关性；
- 在当时可以有效利用多核 CPU；
- advantage 降低 policy gradient 方差；
- 异步更新产生的训练调度对应 A3C 中的 `Asynchronous`；三个 `A` 依次是 `Asynchronous`、`Advantage`、`Actor`。

## 5. A2C：同步版本

OpenAI Baselines 在 2017 年对 A2C 的说明中，将其描述为 A3C 的同步、确定性变体：多个 actor 各收集固定长度片段，等待全部完成后，对其梯度/经验统一聚合并更新。

```text
A3C：worker 完成就异步更新，顺序不固定
A2C：全部 worker 收集完成 -> 聚合 batch -> 同步更新
```

在现代 GPU 批处理条件下，同步 A2C 更容易充分利用大 batch 计算。OpenAI 当时的实现观察到 A2C 在其基准中不劣于异步版本，并未看到异步噪声带来性能收益。这是实现经验，不应外推为所有任务上的定理。

## 6. 与大模型微调的关系

把文本生成视为序列决策：

| RL 概念 | LLM 后训练对应物 |
| --- | --- |
| 状态 `s_t` | prompt 加上已生成 token 前缀 |
| 动作 `a_t` | 下一个 token |
| Actor | 正在优化的语言模型 policy |
| 奖励 | 人类偏好 RM 分数、规则检查器或可验证正确性 |
| Critic | 预测当前前缀未来回报的 value model |

RLHF 的 PPO 流程通常含有 value/critic，因此 Actor-Critic 是理解 PPO 的概念基础。但不宜说 A2C/A3C 本身就是大模型 RLHF 的常规训练算法；现代 LLM RL 实作更常讨论 PPO、RLOO、GRPO 等算法。

GRPO 的一个主要动机，正是通过同一 prompt 下一组回答的相对奖励作为 baseline，去掉单独训练的 Critic/value model。

## 7. 该如何学

### 第一阶段：只理解术语

1. 掌握 `policy`、`value`、`return`、`advantage`、`entropy`。
2. 用 CartPole 之类小环境理解 Actor 与 Critic 分工。
3. 比较 REINFORCE 与 Actor-Critic 的梯度方差直觉。

### 第二阶段：连接 PPO

1. 学习 GAE 如何形成更平滑的 advantage 估计。
2. 再阅读 PPO 的 clipped objective。
3. 最后映射到 LLM：每个 token 的 log probability、sequence reward、KL 约束。

## 8. 推荐阅读资料

### 必读

1. Mnih et al., 2016, *Asynchronous Methods for Deep Reinforcement Learning*（A3C 原始论文，PMLR/ICML）。  
   链接：https://proceedings.mlr.press/v48/mniha16.html
2. OpenAI, 2017, *OpenAI Baselines: ACKTR & A2C*。重点阅读 “A2C and A3C” 小节，对同步/异步区别的工程解释很清楚。  
   链接：https://openai.com/index/openai-baselines-acktr-a2c/

### 补基础

3. Sutton and Barto, *Reinforcement Learning: An Introduction (2nd ed.)*，重点看 policy gradient 与 actor-critic 章节。  
   链接：http://incompleteideas.net/book/the-book-2nd.html
4. OpenAI Spinning Up, *Key Concepts in RL* 与 *Vanilla Policy Gradient*。适合先建立公式直觉。  
   链接：https://spinningup.openai.com/en/latest/spinningup/rl_intro.html  
   链接：https://spinningup.openai.com/en/latest/algorithms/vpg.html

## 9. 阅读完成后的自测题

1. Critic 的 value estimate 为什么能降低 policy gradient 方差？
2. A3C 的三个 `A` 分别是什么？
3. A2C 与 A3C 的主要差异是损失函数还是更新调度？
4. 在 LLM 场景中，状态、动作和奖励分别如何定义？
5. GRPO 去掉 value model 后，用什么信息构造 baseline/advantage？
