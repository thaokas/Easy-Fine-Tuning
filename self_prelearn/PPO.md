# PPO：近端策略优化学习笔记

> 适用目标：理解经典 RLHF 中如何用奖励模型更新语言模型，以及 PPO 与 DPO、GRPO 的差异。  
> 资料核验日期：2026-05-24

## 1. 一句话理解

PPO（Proximal Policy Optimization）是一种 on-policy 强化学习算法：它允许策略根据新采样结果更新，但用 clipped objective 或 KL penalty 限制每次更新不要偏离旧策略太远，从而兼顾学习效果与稳定性。

Schulman et al. 于 2017 年提出 PPO。InstructGPT 将 PPO 用在“奖励模型评分 + KL 约束”的语言模型人类反馈强化学习中，使它成为理解经典 RLHF 的关键算法。

## 2. 从 policy gradient 到 PPO

策略梯度希望增加高优势动作的概率、降低低优势动作的概率。若当前新策略为 `pi_theta`，采样时的旧策略为 `pi_old`，定义概率比：

$$
r_t(\theta) =
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

普通 surrogate objective 是：

$$
L^{PG}(\theta)=
\mathbb{E}_t[r_t(\theta)\hat{A}_t]
$$

但一次更新若把概率比例推得过远，会导致策略突然崩坏。PPO-Clip 使用：

$$
L^{CLIP}(\theta)=
\mathbb{E}_t[
\min(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
)]
$$

直觉：

- 若一个动作比预期好，允许提高概率，但超出一定范围就不再因它获得额外收益；
- 若动作比预期差，允许降低概率，也限制一步变化幅度；
- 它是对“不要从采样策略跳得太远”的简单可实现近似。

## 3. Advantage 与 GAE

PPO 通常与 Actor-Critic 以及 GAE（Generalized Advantage Estimation）一起使用。TD 残差：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

GAE 优势估计：

$$
\hat{A}_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

`lambda` 控制偏差与方差的权衡。对公开课而言，掌握直觉即可：Critic 估计“通常能得多少分”，advantage 则衡量本次输出是否比基线更好。

## 4. PPO 在 LLM RLHF 中如何工作

经典的 RLHF/PPO 流程可以写为：

```text
1. SFT policy 接收 prompts，生成回答
2. reward model 对回答打偏好分
3. reference policy 提供 KL 约束参照
4. value model 预测回报，用于 advantage
5. PPO 更新 policy（并训练 value model）
```

LLM 中的典型总奖励包含偏好奖励及对参考策略的偏离惩罚：

$$
R(x,y)=r_\phi(x,y)-\beta
\log \frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
$$

KL 约束非常重要：只最大化 reward model 分数可能导致 reward hacking，生成偏离自然、有用语言的输出。

### 训练时涉及的组件

| 组件 | 作用 | 通常是否更新 |
| --- | --- | --- |
| Policy / Actor | 生成回答，是最终目标模型 | 是 |
| Reference policy | 提供 KL 参照 | 否 |
| Reward model | 对整段回答评分 | 否 |
| Value / Critic | 预测 return，计算 advantage | 是 |

这也解释了 PPO 式 RLHF 为何比 SFT 或 DPO 更复杂、更吃内存。

## 5. PPO、DPO、GRPO 不要混为一谈

| 方法 | 是否在线采样 | 是否需要显式 value/critic | 奖励/监督信号 |
| --- | --- | --- | --- |
| PPO-RLHF | 是 | 通常需要 | reward model + KL |
| DPO | 否，训练于固定偏好对 | 不需要 | chosen/rejected 比较 |
| GRPO | 是 | 不需要独立 critic | 一组输出的相对奖励，常结合可验证奖励 |

### 适用判断

- 偏好数据已经足够好，希望流程简单稳定：先理解并考虑 DPO。
- 必须通过在线探索持续优化复杂奖励：PPO 仍具有意义。
- 数学/代码等可自动验收推理任务，需要多答案探索：GRPO 是更相关的阅读方向。

因此，“DPO 已全面替代 PPO”是不准确的；它在许多离线偏好对齐任务上更简洁，但不能替代 PPO 的在线探索能力。

## 6. 学习时要注意的难点

1. **on-policy 数据成本**：策略变化后要不断重新采样。
2. **reward hacking**：奖励模型可能被投机性输出欺骗。
3. **KL 与 reward 的平衡**：过强则学不动，过弱则漂移严重。
4. **长度效应**：token 级 KL、sequence reward 和回答长度之间会相互影响。
5. **实现复杂度**：生成、打分、reference logprob、value、policy update 共同影响内存和吞吐。

## 7. 建议阅读与动手路线

1. 先用 Spinning Up 理解 PPO-Clip 的公式图解，不急于进入 LLM。
2. 阅读 InstructGPT 的方法图和 RLHF 阶段，理解 RM、reference 与 PPO 的分工。
3. 阅读 TRL 的 PPOTrainer 示例，识别 policy、ref model、reward model、value model 在代码中的位置。
4. 再阅读 DPO 和 GRPO，制作三者组件数量、采样方式、数据来源的比较图。

## 8. 推荐阅读资料

### 必读

1. Schulman et al., 2017, *Proximal Policy Optimization Algorithms*（PPO 原论文）。  
   链接：https://arxiv.org/abs/1707.06347
2. OpenAI Spinning Up, *Proximal Policy Optimization*。公式与伪代码导读清晰，适合第一次阅读。  
   链接：https://spinningup.openai.com/en/latest/algorithms/ppo.html
3. Ouyang et al., 2022, *Training language models to follow instructions with human feedback*（InstructGPT）。阅读其 PPO-ptx/RLHF 管线。  
   链接：https://arxiv.org/abs/2203.02155

### 实作参考

4. Hugging Face TRL, *PPO Trainer* 官方文档。  
   链接：https://huggingface.co/docs/trl/main/en/ppo_trainer
5. Huang et al., 2024, *The N+ Implementation Details of RLHF with PPO* / Hugging Face blog，可用于理解工程细节，但应与论文和实际版本配置交叉核验。  
   链接：https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo

## 9. 阅读完成后的自测题

1. PPO 为什么要比较 `pi_theta` 和 `pi_old` 的概率比例？
2. clip 对正 advantage 和负 advantage 各有什么限制效果？
3. 在 RLHF 中 reference policy 和 value model 分别承担什么任务？
4. KL 系数过大或过小分别可能导致什么结果？
5. 哪类问题上 DPO 无法提供 PPO 的在线探索特性？
