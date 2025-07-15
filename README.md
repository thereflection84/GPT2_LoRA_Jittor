# 🤖 GPT2_LoRA_Jittor

本项目基于 [Jittor](https://github.com/Jittor/jittor) 框架，在经典英文问答数据集 [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) 上对 [GPT2](https://github.com/openai/gpt-2) 进行 LoRA 微调，在部分数据集的情况下验证其理解能力复现并实现了 GPT2 模型的 LoRA（Low-Rank Adaptation）微调方法，支持中文与英文任务的训练与推理。

> 🔬 本项目旨在验证 LoRA 微调技术在 Jittor 框架下的可行性与高效性，提供完整的训练、推理和可视化支持。

---

## 📁 项目结构

```
jittor_lora_gpt2/
├── GPT2_ji.py                   # GPT2 模型结构（Block, Attention, Feedforward 等）
├── LoRA.py                      # LoRA 插入模块，支持低秩适配
├── tokenizer_loader.py          # Tokenizer 加载模块（兼容 Huggingface）
├── dataset_loader.py            # 数据集预处理与加载（支持中英文）
├── GPT2_LoRA_Full_Experiment.py # 主训练脚本（支持训练与测试）
├── plot_loss.py                 # 绘制训练过程损失曲线
├── config/                      # 可选的模型配置文件目录
├── checkpoints/                 # 保存模型权重的目录
└── README.md
```

---

## 🔧 环境依赖

- Python ≥ 3.8
- Jittor ≥ 1.3.7.0（建议 GPU 版本）
- Transformers（用于加载 GPT2 tokenizer 和预训练权重）
- 其他依赖：`tqdm`, `numpy`, `matplotlib`

### 安装命令

```bash
pip install jittor==1.3.7.0
pip install transformers==4.30.0
pip install matplotlib tqdm numpy
```

> 💡 请确保你正确安装了 GPU 版 Jittor，参考官网安装说明：[https://cg.cs.tsinghua.edu.cn/jittor/install](https://cg.cs.tsinghua.edu.cn/jittor/install)

---

## 🚀 快速开始

### 1️⃣ 准备 GPT2 Tokenizer 与预训练模型

```python
from transformers import GPT2Tokenizer, GPT2Model
GPT2Tokenizer.from_pretrained("gpt2").save_pretrained("./gpt2")
GPT2Model.from_pretrained("gpt2").save_pretrained("./gpt2")
```

### 2️⃣ 运行微调脚本

```bash
python GPT2_LoRA_Full_Experiment.py
```

### 3️⃣ 查看生成文本

```
输入提示词：人工智能
生成结果：人工智能正在改变人类的生活方式，在医疗、教育和交通等领域展现巨大潜力……
```

---

## 🧠 LoRA 模块机制简介

LoRA 是一种低秩矩阵近似微调方法，可减少参数更新量并加快训练速度。本项目中 LoRA 应用于 GPT2 的 Attention 子层。

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=32):
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
    def execute(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling
```

注入方式：

```python
# 替换原始 Q/K/V 层为 LoRA 注入层
self.q_proj = LoRAInjectedLinear(original_q_proj)
```

---

## 📊 实验结果示例

| Epoch | Loss  | Perplexity |
|-------|-------|------------|
|   1   | 2.63  | 13.9       |
|   2   | 1.95  | 7.03       |
|   3   | 1.52  | 4.58       |

### 📈 损失曲线可视化

保存训练时损失的 loss.txt 文件后运行：

```bash
python plot_loss.py
```

输出图示：

![loss curve](./images/loss_curve.png)

---

## 📚 参考资料

- 🔖 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- 🔖 [Jittor: A Novel Deep Learning Framework](https://github.com/Jittor/jittor)
- 🔖 [Huggingface Transformers](https://huggingface.co/docs/transformers)

---

## 📎 引用本项目

如果你在学术或工程项目中使用了本项目，请引用：

```bibtex
@misc{jittor-lora-gpt2,
  author = {Your Name},
  title = {Jittor Implementation of LoRA on GPT2},
  year = {2025},
  howpublished = {\url{https://github.com/yourname/jittor-lora-gpt2}}
}
```

---

## 🤝 贡献方式

欢迎贡献代码、改进文档或报告问题：

```bash
# fork 仓库后提交 PR
git clone https://github.com/yourname/jittor-lora-gpt2.git
```

欢迎 star ⭐ 本项目以示鼓励！
```
