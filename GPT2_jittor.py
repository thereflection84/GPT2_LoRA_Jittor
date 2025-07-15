import math
import jittor as jt
import jittor.nn as nn
import numpy as np
from dataclasses import dataclass
import os

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 词嵌入
        self.transformer_wte = nn.Embedding(config.vocab_size, config.n_embed)
        # 位置嵌入
        self.transformer_wpe = nn.Embedding(config.block_size, config.n_embed)
        # Transformer层
        self.transformer_h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # 最终层归一化
        self.transformer_ln_f = nn.LayerNorm(config.n_embed)
        # 语言模型头
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # 权重共享
        self.lm_head.weight = self.transformer_wte.weight

    def execute(self, x, targets=None):
        # 获取批次大小和序列长度
        B, T = x.shape
        
        # 获取位置编码
        pos = jt.arange(0, T)
        pos_emb = self.transformer_wpe(pos)
        
        # 获取词嵌入并添加位置编码
        tok_emb = self.transformer_wte(x)
        x = tok_emb + pos_emb
        
        # 通过所有Transformer层
        for block in self.transformer_h:
            x = block(x)
            
        # 最终层归一化
        x = self.transformer_ln_f(x)
        
        # 输出层
        logits = self.lm_head(x)
        
        # 计算损失（如果有目标）
        if targets is not None:
            loss = nn.cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            return logits, loss
        else:
            return logits
            
    @classmethod
    def from_pretrained(cls, model_path):
        """从预训练权重加载模型"""
        from transformers import GPT2Config
        
        print(f"从本地加载模型权重: {model_path}")
        
        # 加载配置
        hf_config = GPT2Config.from_pretrained(model_path)
        config = GPTConfig(
            block_size=hf_config.n_positions,
            vocab_size=hf_config.vocab_size,
            n_layer=hf_config.n_layer,
            n_head=hf_config.n_head,
            n_embed=hf_config.n_embd
        )
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        npz_path = os.path.join(os.path.dirname(model_path), "gpt2_weights.npz")
        if not os.path.exists(npz_path):
            print(f"未找到npz权重文件，请先运行 loadweight.py 转换权重")
            return model
            
        print(f"加载npz权重文件: {npz_path}")
        weights = np.load(npz_path)
        
        # 创建权重映射
        mapping = {}
        
        # 嵌入层和最终层归一化
        mapping["transformer.wte.weight"] = "transformer_wte.weight"
        mapping["transformer.wpe.weight"] = "transformer_wpe.weight"
        mapping["transformer.ln_f.weight"] = "transformer_ln_f.weight"
        mapping["transformer.ln_f.bias"] = "transformer_ln_f.bias"
        
        # Transformer层
        for i in range(config.n_layer):
            # 第一个归一化层
            mapping[f"transformer.h.{i}.ln_1.weight"] = f"transformer_h.{i}.ln_1.weight"
            mapping[f"transformer.h.{i}.ln_1.bias"] = f"transformer_h.{i}.ln_1.bias"
            
            # 注意力层
            mapping[f"transformer.h.{i}.attn.c_attn.weight"] = f"transformer_h.{i}.attn.c_attn.weight"
            mapping[f"transformer.h.{i}.attn.c_attn.bias"] = f"transformer_h.{i}.attn.c_attn.bias"
            mapping[f"transformer.h.{i}.attn.c_proj.weight"] = f"transformer_h.{i}.attn.c_proj.weight"
            mapping[f"transformer.h.{i}.attn.c_proj.bias"] = f"transformer_h.{i}.attn.c_proj.bias"
            
            # 第二个归一化层
            mapping[f"transformer.h.{i}.ln_2.weight"] = f"transformer_h.{i}.ln_2.weight"
            mapping[f"transformer.h.{i}.ln_2.bias"] = f"transformer_h.{i}.ln_2.bias"
            
            # MLP层
            mapping[f"transformer.h.{i}.mlp.c_fc.weight"] = f"transformer_h.{i}.mlp.c_fc.weight"
            mapping[f"transformer.h.{i}.mlp.c_fc.bias"] = f"transformer_h.{i}.mlp.c_fc.bias"
            mapping[f"transformer.h.{i}.mlp.c_proj.weight"] = f"transformer_h.{i}.mlp.c_proj.weight"
            mapping[f"transformer.h.{i}.mlp.c_proj.bias"] = f"transformer_h.{i}.mlp.c_proj.bias"
        
        # 需要转置的层
        transpose_layers = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        
        # 加载权重
        loaded = 0
        for pt_name, jt_name in mapping.items():
            if pt_name in weights:
                try:
                    weight = weights[pt_name]
                    
                    # 检查是否需要转置
                    if any(layer in jt_name for layer in transpose_layers):
                        weight = weight.T
                    
                    # 直接处理参数路径
                    if "transformer_h" in jt_name:
                        # 例如: transformer_h.0.ln_1.weight
                        parts = jt_name.split('.')
                        layer_idx = int(parts[1])
                        sublayer = '.'.join(parts[2:])
                        
                        # 直接访问ModuleList中的元素
                        if hasattr(model.transformer_h[layer_idx], parts[2]):
                            if len(parts) == 4:  # 例如 transformer_h.0.ln_1.weight
                                param = getattr(model.transformer_h[layer_idx], parts[2]).weight if parts[3] == "weight" else getattr(model.transformer_h[layer_idx], parts[2]).bias
                            elif len(parts) == 5:  # 例如 transformer_h.0.attn.c_attn.weight
                                component = getattr(model.transformer_h[layer_idx], parts[2])
                                param = getattr(component, parts[3]).weight if parts[4] == "weight" else getattr(component, parts[3]).bias
                            else:
                                print(f"无法处理的参数路径: {jt_name}")
                                continue
                        else:
                            print(f"模块不存在: {parts[2]} in layer {layer_idx}")
                            continue
                    else:
                        # 处理非transformer_h的参数
                        parts = jt_name.split('.')
                        if len(parts) == 2:  # 例如 transformer_wte.weight
                            param = getattr(model, parts[0]).weight if parts[1] == "weight" else getattr(model, parts[0]).bias
                        else:
                            print(f"无法处理的参数路径: {jt_name}")
                            continue
                    
                    # 赋值
                    param.assign(jt.array(weight))
                    loaded += 1
                except Exception as e:
                    print(f"加载权重 {pt_name} 失败: {e}")
        
        print(f"成功加载 {loaded}/{len(mapping)} 个参数")
        return model


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def execute(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        
        # QKV投影
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        
        # 保存配置
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.block_size = config.block_size
        
        # 创建因果mask - 使用numpy创建下三角矩阵，避免jt.tril的参数问题
        mask = np.tril(np.ones((config.block_size, config.block_size)))
        # 在Jittor中register_buffer只接受一个参数，需要手动设置属性
        self.mask = jt.array(mask)

    def execute(self, x):
        # 使用numpy处理大部分计算，避免Jittor API兼容性问题
        B, T, C = x.shape
        
        # QKV投影
        qkv = self.c_attn(x).numpy()
        
        # 分割QKV
        q, k, v = np.split(qkv, 3, axis=2)
        
        # 多头注意力
        head_size = C // self.n_head
        
        # 重塑为多头形式
        q = q.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        
        # 计算注意力分数
        att = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(head_size)  # (B, nh, T, T)
        
        # 应用因果mask
        mask_np = self.mask.numpy()[:T, :T]
        causal_mask = np.tril(np.ones((T, T)))
        
        # 为每个批次和头部应用mask
        for b in range(B):
            for h in range(self.n_head):
                att[b, h] = np.where(causal_mask == 0, -1e10, att[b, h])
        
        # 应用softmax
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        att = att / (np.sum(att, axis=-1, keepdims=True) + 1e-10)
        
        # 加权聚合
        out = np.matmul(att, v)  # (B, nh, T, hs)
        
        # 重塑回原始形状
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 输出投影
        return self.c_proj(jt.array(out))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.act = nn.GELU()

    def execute(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    """使用模型生成文本"""
    model.eval()
    
    # 编码提示文本
    input_ids = tokenizer.encode(prompt)
    
    # 生成文本
    for _ in range(max_new_tokens):
        # 准备输入
        inputs = jt.array(input_ids).reshape(1, -1)
        
        # 获取预测
        with jt.no_grad():
            outputs = model(inputs)
            
        # 获取最后一个token的预测
        next_token_logits = outputs[0, -1, :].numpy() / temperature
        
        # Top-K采样
        indices = np.argsort(-next_token_logits)
        top_k_indices = indices[:top_k]
        top_k_logits = next_token_logits[top_k_indices]
        
        # 计算概率
        top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        # 采样
        idx = np.random.choice(top_k_indices, p=top_k_probs)
        
        # 添加到序列
        input_ids.append(int(idx))
        
        # 如果生成了结束符，则停止
        if idx == tokenizer.eos_token_id:
            break
    
    # 解码生成的文本
    return tokenizer.decode(input_ids)
