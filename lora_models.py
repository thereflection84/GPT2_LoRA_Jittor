import os
import numpy as np
import jittor as jt
import jittor.nn as nn
from GPT2_jittor import Block, CausalSelfAttention, MLP
from LoRA import LoRALinear, LoRAQKVProjection, LoRAConfig


# LoRA版本的Block
class LoRABlock(nn.Module):
    def __init__(self, config, lora_config, original_block=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        
        # 获取原始注意力和MLP模块
        original_attn = None
        original_mlp = None
        if original_block is not None:
            self.ln_1 = original_block.ln_1  # 复用原始层归一化
            original_attn = original_block.attn
            original_mlp = original_block.mlp
            self.ln_2 = original_block.ln_2  # 复用原始层归一化
        else:
            self.ln_2 = nn.LayerNorm(config.n_embed)
        
        # 根据target_modules配置决定是否对注意力机制应用LoRA
        if "c_attn" in lora_config.target_modules or "c_proj" in lora_config.target_modules:
            # 如果有原始注意力模块，提取其中的c_attn和c_proj
            original_c_attn = getattr(original_attn, 'c_attn', None) if original_attn else None
            original_c_proj = getattr(original_attn, 'c_proj', None) if original_attn else None
            
            self.attn = LoRACausalSelfAttention(
                config, 
                lora_config, 
                original_c_attn=original_c_attn,
                original_c_proj=original_c_proj
            )
        else:
            self.attn = original_attn if original_attn else CausalSelfAttention(config)
            
        # 根据target_modules配置决定是否对MLP应用LoRA
        if "c_fc" in lora_config.target_modules or "c_proj" in lora_config.target_modules:
            # 如果有原始MLP模块，提取其中的c_fc和c_proj
            original_c_fc = getattr(original_mlp, 'c_fc', None) if original_mlp else None
            original_c_proj = getattr(original_mlp, 'c_proj', None) if original_mlp else None
            
            self.mlp = LoRAMLP(
                config, 
                lora_config,
                original_c_fc=original_c_fc,
                original_c_proj=original_c_proj
            )
        else:
            self.mlp = original_mlp if original_mlp else MLP(config)

    def execute(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# LoRA版本的注意力模块
class LoRACausalSelfAttention(nn.Module):
    def __init__(self, config, lora_config, original_c_attn=None, original_c_proj=None):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        
        # 保存配置
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.block_size = config.block_size
        self.head_size = config.n_embed // config.n_head
        
        # 根据target_modules配置决定是否对c_attn应用LoRA
        if "c_attn" in lora_config.target_modules:
            original_weights = None
            if original_c_attn is not None:
                original_weights = original_c_attn.weight
            
            # 根据qv_only配置决定对哪些部分应用LoRA
            apply_q = True  # 总是应用于Q
            apply_k = not lora_config.qv_only  # 只在非qv_only模式下应用于K
            apply_v = True  # 总是应用于V
            
            # 创建分离的QKV投影层
            self.qkv_proj = LoRAQKVProjection(
                hidden_size=config.n_embed,
                head_size=self.head_size,
                num_heads=config.n_head,
                apply_q=apply_q,
                apply_k=apply_k,
                apply_v=apply_v,
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                original_weights=original_weights
            )
            
            # 不再需要c_attn，直接使用qkv_proj
            self.use_qkv_proj = True
        else:
            self.c_attn = original_c_attn if original_c_attn is not None else nn.Linear(config.n_embed, 3 * config.n_embed)
            self.use_qkv_proj = False
        
        # 根据target_modules配置决定是否对c_proj应用LoRA
        if "c_proj" in lora_config.target_modules:
            # 获取原始权重和偏置
            original_weights = None
            original_bias = None
            if original_c_proj is not None:
                original_weights = original_c_proj.weight
                original_bias = original_c_proj.bias
                
            self.c_proj = LoRALinear(
                config.n_embed,
                config.n_embed,
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                original_weights=original_weights,
                original_bias=original_bias
            )
        else:
            self.c_proj = original_c_proj if original_c_proj is not None else nn.Linear(config.n_embed, config.n_embed)
        
        # 创建因果mask
        mask = np.tril(np.ones((config.block_size, config.block_size)))
        self.register_buffer("mask", jt.array(mask))

    def execute(self, x):
        # 尽量使用Jittor操作，避免NumPy转换
        B, T, C = x.shape
        
        # QKV投影 - 使用不同的方法根据是否使用分离的QKV投影
        if self.use_qkv_proj:
            # 使用分离的QKV投影
            q, k, v = self.qkv_proj(x)
        else:
            # 使用原始的c_attn
            qkv = self.c_attn(x)
            
            # 分割QKV
            q, k, v = jt.chunk(qkv, 3, dim=2)
            
            # 多头注意力处理
            q = q.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
        
        # 计算注意力分数 - 使用Jittor操作
        # q, k, v形状: (B, nh, T, hs)
        att = jt.matmul(q, k.transpose(0, 1, 3, 2)) / jt.sqrt(jt.float32(self.head_size))
        
        # 应用因果mask
        # 截取当前序列长度的mask
        mask = self.mask[:T, :T]
        # 应用mask (B, nh, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax
        att = jt.nn.softmax(att, dim=-1)
        
        # 加权聚合
        out = jt.matmul(att, v)  # (B, nh, T, hs)
        
        # 重塑回原始形状
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 输出投影
        return self.c_proj(out)


# LoRA版本的MLP
class LoRAMLP(nn.Module):
    def __init__(self, config, lora_config, original_c_fc=None, original_c_proj=None):
        super().__init__()
        # 根据target_modules配置决定是否对c_fc应用LoRA
        if "c_fc" in lora_config.target_modules:
            # 获取原始权重和偏置
            original_weights = None
            original_bias = None
            if original_c_fc is not None:
                original_weights = original_c_fc.weight
                original_bias = original_c_fc.bias
                
            self.c_fc = LoRALinear(
                config.n_embed, 
                4 * config.n_embed,
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                original_weights=original_weights,
                original_bias=original_bias
            )
        else:
            self.c_fc = original_c_fc if original_c_fc is not None else nn.Linear(config.n_embed, 4 * config.n_embed)
            
        # 根据target_modules配置决定是否对c_proj应用LoRA
        if "c_proj" in lora_config.target_modules:
            # 获取原始权重和偏置
            original_weights = None
            original_bias = None
            if original_c_proj is not None:
                original_weights = original_c_proj.weight
                original_bias = original_c_proj.bias
                
            self.c_proj = LoRALinear(
                4 * config.n_embed, 
                config.n_embed,
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                original_weights=original_weights,
                original_bias=original_bias
            )
        else:
            self.c_proj = original_c_proj if original_c_proj is not None else nn.Linear(4 * config.n_embed, config.n_embed)
            
        self.act = nn.GELU()

    def execute(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


# LoRA版本的GPT模型
class LoRAGPT(nn.Module):
    def __init__(self, base_model, lora_config):
        super().__init__()
        self.config = base_model.config
        
        # 保留原始嵌入层
        self.transformer_wte = base_model.transformer_wte
        self.transformer_wpe = base_model.transformer_wpe
        
        # 创建LoRA版本的Transformer层，传递原始块
        self.transformer_h = nn.ModuleList()
        for i in range(self.config.n_layer):
            # 获取对应的原始块
            original_block = base_model.transformer_h[i] if hasattr(base_model, 'transformer_h') else None
            self.transformer_h.append(LoRABlock(self.config, lora_config, original_block))
        
        # 保留原始层归一化和语言模型头
        self.transformer_ln_f = base_model.transformer_ln_f
        self.lm_head = base_model.lm_head
        
        # 记录哪些参数是可训练的
        self.trainable_param_names = []
        for name, _ in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                self.trainable_param_names.append(name)
        
    def get_trainable_params(self):
        """获取可训练的参数数量"""
        trainable_params = 0
        all_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if name in self.trainable_param_names:
                trainable_params += num_params
        return trainable_params, all_params
    
    def print_trainable_parameters(self):
        """打印可训练参数的比例"""
        trainable_params, all_params = self.get_trainable_params()
        print(f"可训练参数: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"所有参数: {all_params:,}")
        print(f"可训练参数名称: {self.trainable_param_names}")

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
    
    def save_lora_weights(self, save_path):
        """仅保存LoRA权重"""
        lora_state_dict = {}
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_state_dict[name] = param.numpy()
        
        # 保存为npz格式
        np.savez(save_path, **lora_state_dict)
        print(f"LoRA权重已保存至 {save_path}")
        
    def load_lora_weights(self, load_path):
        """加载LoRA权重"""
        if not os.path.exists(load_path):
            print(f"未找到LoRA权重文件: {load_path}")
            return False
            
        weights = np.load(load_path)
        loaded = 0
        
        for name, param in self.named_parameters():
            if name in weights:
                try:
                    param.assign(jt.array(weights[name]))
                    loaded += 1
                except Exception as e:
                    print(f"加载权重 {name} 失败: {e}")
        
        print(f"成功加载 {loaded} 个LoRA参数")
        return True 
