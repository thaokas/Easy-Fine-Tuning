# Fine-Tuning

本仓库整理了大模型微调课程资料与一个精简后的 MiniMind 训练作业目录。根目录主要包含：

- `documents/`：课程讲义、PPT 与相关技术笔记。
- `homework/`：基于 MiniMind 的模型结构与训练脚本，用于预训练、SFT、LoRA、DPO、PPO、GRPO 等作业实验。

## documents 文件说明

| 路径 | 简短说明 |
| --- | --- |
| `documents/课程内容.md` | 课程主体内容整理，覆盖微调与强化学习相关知识点。 |
| `documents/outline.md` | 课程/汇报提纲，用于梳理整体结构。 |
| `documents/rl.md` | 强化学习与大模型对齐相关笔记。 |
| `documents/PPT.html` | 课程展示用 HTML 版幻灯片。 |
| `documents/PPT.pdf` | 课程展示用 PDF 版幻灯片。 |
| `documents/related_intro/SFT.md` | SFT 监督微调简介。 |
| `documents/related_intro/LoRA.md` | LoRA 参数高效微调简介。 |
| `documents/related_intro/DPO.md` | DPO 偏好优化简介。 |
| `documents/related_intro/PPO.md` | PPO 强化学习算法简介。 |
| `documents/related_intro/GRPO.md` | GRPO 组相对策略优化简介。 |
| `documents/related_intro/Actor-Critic_A2C_A3C.md` | Actor-Critic、A2C、A3C 算法简介。 |

## homework 如何运行

`homework/README.md` 是 MiniMind 原项目 README，包含大量完整项目说明；但当前 `homework/` 是课程作业用的精简目录，主要保留了模型、数据集封装和训练脚本。因此旧 README 中关于以下内容的命令在当前目录下不能直接使用：

- `eval_llm.py` 推理脚本。
- `scripts/` 下的 WebUI、OpenAI API、模型转换脚本。
- 第三方推理框架部署、完整模型发布与在线体验等完整项目流程。

### 1. 安装依赖

进入 `homework` 目录，创建环境并安装依赖：

```bash
cd homework
pip install -r requirements.txt
```

`requirements.txt` 中没有固定安装 `torch`，需要根据本机 CUDA/CPU 环境单独安装合适版本的 PyTorch。

### 2. 准备数据、tokenizer 与权重

当前目录已经包含 `homework/dataset/lm_dataset.py`、`homework/model/` 和 `homework/trainer/`，但不包含实际训练数据、tokenizer 文件和模型权重。运行前需要按需准备：

- 将下载的数据文件放到 `homework/dataset/`，例如 `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl`、`dpo.jsonl`、`rlaif.jsonl`、`lora_medical.jsonl`。minimind项目将数据放在了 `huggingface` 和 `modelscope` 的 `minimind-3` 数据集上，需要自行下载。
- 将 tokenizer 相关文件放到 `homework/model/`；`trainer_utils.py` 默认通过 `AutoTokenizer.from_pretrained("../model")` 加载。
- 若执行 SFT、LoRA、DPO、PPO、GRPO 等后续阶段，需要在 `homework/out/` 中准备对应前置权重，例如 `pretrain_768.pth` 或 `full_sft_768.pth`。
- PPO / GRPO 还需要 reward model，脚本默认路径为 `../../internlm2-1_8b-reward`，也可通过 `--reward_model_path` 指定。


### 3. 训练命令

以下命令需要在 `homework/trainer` 目录下执行。

预训练：

```bash
cd homework/trainer
python train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl
```

全参数 SFT：

```bash
python train_full_sft.py --data_path ../dataset/sft_t2t_mini.jsonl --from_weight pretrain
```

LoRA 微调：

```bash
python train_lora.py --data_path ../dataset/lora_medical.jsonl --from_weight full_sft
```

DPO：

```bash
python train_dpo.py --data_path ../dataset/dpo.jsonl --from_weight full_sft
```

PPO / GRPO：

```bash
python train_ppo.py --data_path ../dataset/rlaif.jsonl --from_weight full_sft --rollout_engine torch
python train_grpo.py --data_path ../dataset/rlaif.jsonl --from_weight full_sft --rollout_engine torch
```

训练输出默认保存到 `homework/out/`，断点续训文件默认保存到 `homework/checkpoints/`。如需从断点恢复，可追加：

```bash
--from_resume 1
```

### 4. 多卡训练

训练脚本支持 PyTorch DDP。若有 `N` 张 GPU，可在 `homework/trainer` 下执行：

```bash
torchrun --nproc_per_node N train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl
```

把 `train_pretrain.py` 替换为其他训练脚本即可运行对应阶段。

## 许可与第三方声明

- 根目录 `LICENSE` 与 `THIRD_PARTY_NOTICES.md` 记录本仓库整体许可和第三方声明。
- `homework/LICENSE` 与 `homework/THIRD_PARTY_NOTICE.md` 记录作业目录所引用 MiniMind 相关内容的许可与声明。
