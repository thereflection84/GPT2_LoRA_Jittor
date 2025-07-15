import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass
import os
from transformers import GPT2Config, GPT2LMHeadModel


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

        # 设定词嵌入矩阵与分类器权重共享
        self.lm_head.weight = self.transformer_wte.weight

    def forward(self, x, targets=None):
        # 获取批次大小和序列长度
        B, T = x.shape
        
        # 获取位置编码
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            return logits
            
    @classmethod
    def from_pretrained(cls, model_path):
        """从预训练权重加载模型"""
        print(f"从预训练模型加载: {model_path}")
        
        # 使用transformers库加载GPT2模型
        hf_model = GPT2LMHeadModel.from_pretrained(model_path)
        hf_config = hf_model.config
        
        # 创建我们的配置
        config = GPTConfig(
            block_size=hf_config.n_positions,
            vocab_size=hf_config.vocab_size,
            n_layer=hf_config.n_layer,
            n_head=hf_config.n_head,
            n_embed=hf_config.n_embd
        )
        
        # 创建我们的模型
        model = cls(config)
        
        # 复制权重
        # 嵌入层
        model.transformer_wte.weight.data.copy_(hf_model.transformer.wte.weight.data)
        model.transformer_wpe.weight.data.copy_(hf_model.transformer.wpe.weight.data)
        
        # Transformer层
        for i in range(config.n_layer):
            # 注意力层
            # 检查权重维度，确保正确处理
            hf_c_attn_weight = hf_model.transformer.h[i].attn.c_attn.weight.data
            hf_c_attn_bias = hf_model.transformer.h[i].attn.c_attn.bias.data
            
            # 打印权重维度以便调试
            print(f"Layer {i} c_attn weight shape: model={model.transformer_h[i].attn.c_attn.weight.shape}, hf={hf_c_attn_weight.shape}")
            
            # 确保维度匹配 - 需要转置HuggingFace的权重
            # HuggingFace使用[in_features, out_features]格式，而PyTorch标准是[out_features, in_features]
            hf_c_attn_weight_transposed = hf_c_attn_weight.t()  # 转置权重
            print(f"转置后权重形状: {hf_c_attn_weight_transposed.shape}")
            
            # 确保维度匹配
            if model.transformer_h[i].attn.c_attn.weight.shape != hf_c_attn_weight_transposed.shape:
                print(f"维度不匹配，尝试调整...")
                # 如果我们的模型使用的是不同的维度，需要调整
                out_features, in_features = hf_c_attn_weight_transposed.shape
                # 重新创建c_attn层以匹配HF模型的维度
                model.transformer_h[i].attn.c_attn = nn.Linear(in_features, out_features)
                print(f"重新创建Layer {i}的c_attn层以匹配维度: {model.transformer_h[i].attn.c_attn.weight.shape}")
            
            # 现在复制转置后的权重
            model.transformer_h[i].attn.c_attn.weight.data.copy_(hf_c_attn_weight_transposed)
            model.transformer_h[i].attn.c_attn.bias.data.copy_(hf_c_attn_bias)
            
            # 其他层的权重复制 - 同样需要转置
            model.transformer_h[i].attn.c_proj.weight.data.copy_(hf_model.transformer.h[i].attn.c_proj.weight.data.t())
            model.transformer_h[i].attn.c_proj.bias.data.copy_(hf_model.transformer.h[i].attn.c_proj.bias.data)
            
            # 层归一化
            model.transformer_h[i].ln_1.weight.data.copy_(hf_model.transformer.h[i].ln_1.weight.data)
            model.transformer_h[i].ln_1.bias.data.copy_(hf_model.transformer.h[i].ln_1.bias.data)
            model.transformer_h[i].ln_2.weight.data.copy_(hf_model.transformer.h[i].ln_2.weight.data)
            model.transformer_h[i].ln_2.bias.data.copy_(hf_model.transformer.h[i].ln_2.bias.data)
            
            # MLP层 - 同样需要转置
            model.transformer_h[i].mlp.c_fc.weight.data.copy_(hf_model.transformer.h[i].mlp.c_fc.weight.data.t())
            model.transformer_h[i].mlp.c_fc.bias.data.copy_(hf_model.transformer.h[i].mlp.c_fc.bias.data)
            model.transformer_h[i].mlp.c_proj.weight.data.copy_(hf_model.transformer.h[i].mlp.c_proj.weight.data.t())
            model.transformer_h[i].mlp.c_proj.bias.data.copy_(hf_model.transformer.h[i].mlp.c_proj.bias.data)
        
        # 最终层归一化
        model.transformer_ln_f.weight.data.copy_(hf_model.transformer.ln_f.weight.data)
        model.transformer_ln_f.bias.data.copy_(hf_model.transformer.ln_f.bias.data)
        
        print(f"成功加载预训练权重")
        return model


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        
        # QKV投影 - 输出维度是输入维度的3倍，因为同时生成Q、K、V
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        
        # 保存配置
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.block_size = config.block_size
        
        # 注册因果mask缓冲区
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.shape
        
        # 计算QKV
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=2)  # 每个都是(B, T, C)
        
        # 重塑为多头形式
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # 加权聚合
        y = att @ v  # (B, nh, T, hs)
        
        # 重塑回原始形状
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # 输出投影
        y = self.c_proj(y)
        
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    """使用模型生成文本"""
    model.eval()
    
    # 编码提示文本
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    
    # 移动到正确的设备
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 如果序列太长，截断
            if input_ids.size(1) > model.config.block_size:
                input_ids = input_ids[:, -model.config.block_size:]
                
            # 获取预测
            outputs = model(input_ids)
            
            # 获取最后一个token的预测
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Top-K采样
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # 计算概率
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # 采样
            idx_next = top_k_indices[torch.multinomial(top_k_probs, num_samples=1)]
            
            # 添加到序列
            input_ids = torch.cat((input_ids, idx_next.unsqueeze(0)), dim=1)
            
            # 如果生成了结束符，则停止
            if idx_next.item() == tokenizer.eos_token_id:
                break
    
    # 解码生成的文本
    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    import torch
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    model = GPT.from_pretrained('gpt2')
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 测试生成文本
    prompts = [
        "Hello, I'm a language model,",
        "The meaning of life is",
        "Once upon a time in a galaxy far, far away",
        "The best way to learn programming is"
    ]
    
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=30)
        print(f"\n输入: {prompt}")
        print(f"输出: {generated}")
        print("-" * 50) 
