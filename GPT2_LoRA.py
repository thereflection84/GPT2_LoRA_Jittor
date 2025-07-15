import math
import os
import json
import time
import numpy as np
import jittor as jt
import jittor.nn as nn
from dataclasses import dataclass, field
from tqdm import tqdm
from transformers import GPT2Tokenizer

# 显式启用GPU
jt.flags.use_cuda = 1

# 导入基础GPT2模型
from GPT2_jittor import GPTConfig, GPT, Block, CausalSelfAttention, MLP


# LoRA配置
@dataclass
class LoRAConfig:
    r: int = 8  # LoRA秩
    alpha: int = 16  # LoRA缩放因子
    dropout: float = 0.1  # LoRA dropout
    target_modules: list = field(default_factory=list)  # 要应用LoRA的模块列表
    qv_only: bool = False  # 是否只对Q和V应用LoRA

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = []


# LoRA线性层
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.1, original_weights=None, original_bias=None):
        super().__init__()

        if original_weights is not None:
            # 使用传入的预训练权重
            self.weight = original_weights
        else:
            # 如果没有提供预训练权重，创建随机权重（仅用于测试）
            self.weight = jt.init.gauss((out_features, in_features), 'float32', mean=0.0, std=0.02)
        
        # 确保原始权重不参与梯度更新
        self.weight.requires_grad = False
        
        # 处理偏置
        if original_bias is not None:
            self.bias = original_bias
            self.bias.requires_grad = False
        else:
            self.bias = jt.zeros(out_features)
            self.bias.requires_grad = False
        
        # LoRA低秩分解
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # 初始化A为随机值，B为零
        # 使用Jittor兼容的初始化方式
        self.lora_A.weight = jt.init.gauss(self.lora_A.weight.shape, 'float32', std=1/r)
        self.lora_B.weight = jt.zeros(self.lora_B.weight.shape)

    def execute(self, x):
        # 原始层的前向传播
        orig_output = jt.matmul(x, self.weight.t())
        if self.bias is not None:
            orig_output += self.bias
            
        # LoRA路径的前向传播
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        
        # 合并输出
        return orig_output + lora_output


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


# 数据加载器
class SQuADDataLoader:
    def __init__(self, data_path, tokenizer, block_size=1024, batch_size=4, shuffle=True):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # 提取训练和验证集
        self.train_data = self.data.get('train', [])
        self.val_data = self.data.get('val', self.train_data[:len(self.train_data)//10])
        
        print(f"加载了 {len(self.train_data)} 个训练样本和 {len(self.val_data)} 个验证样本")
    
    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        
        # 随机选择样本
        indices = np.random.randint(0, len(data), size=self.batch_size)
        
        # 准备批次
        x = []
        y = []
        
        for idx in indices:
            tokens = data[idx]['tokens']
            
            # 确保长度不超过block_size
            if len(tokens) > self.block_size:
                tokens = tokens[:self.block_size]
                
            # 输入和目标（目标是输入向右移动一位）
            x_sample = tokens[:-1]
            y_sample = tokens[1:]
            
            # 填充到相同长度
            if len(x_sample) < self.block_size - 1:
                x_sample = x_sample + [0] * (self.block_size - 1 - len(x_sample))
                y_sample = y_sample + [0] * (self.block_size - 1 - len(y_sample))
            
            x.append(x_sample)
            y.append(y_sample)
        
        # 转换为Jittor张量
        x_tensor = jt.array(np.array(x))
        y_tensor = jt.array(np.array(y))
        
        return x_tensor, y_tensor


# 添加梯度监控函数
def check_grads(model, optimizer, step, verbose=True):
    """检查模型参数梯度，返回有多少百分比的参数有梯度更新"""
    total_params = 0
    params_with_grad = 0
    zero_grad_params = []
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            total_params += 1
            # 在Jittor中，使用param.opt_grad(optimizer)访问梯度
            try:
                grad = param.opt_grad(optimizer)
                has_grad = grad is not None
                if has_grad:
                    params_with_grad += 1
                else:
                    zero_grad_params.append(name)
            except Exception:
                zero_grad_params.append(name)
    
    percent = 100 * params_with_grad / total_params if total_params > 0 else 0
    
    if verbose and step % 10 == 0:  # 每10步打印一次
        print(f"\n步骤 {step}: {params_with_grad}/{total_params} ({percent:.1f}%) LoRA参数有梯度")
        if zero_grad_params and len(zero_grad_params) < 10:
            print(f"无梯度参数: {zero_grad_params}")
    
    return percent

# 训练函数
def train_lora(
    base_model_path='./local_gpt2',
    data_path='./data/processed_squad_100.json',
    output_dir='./checkpoints',
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    batch_size=4,
    learning_rate=5e-4,
    num_epochs=3,
    save_every=100,
    eval_every=50,
    max_grad_norm=1.0,
    target_modules=None,
    lora_config=None,
    debug_gradients=True  # 添加调试梯度的选项
):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 加载基础模型
    base_model = GPT.from_pretrained(base_model_path)
    
    # 创建LoRA配置
    if lora_config is None:
        lora_config = LoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules if target_modules else ["c_attn", "c_proj", "c_fc"]
        )
    elif target_modules and not lora_config.target_modules:
        lora_config.target_modules = target_modules
    
    # 创建LoRA模型
    model = LoRAGPT(base_model, lora_config)
    model.print_trainable_parameters()
    
    # 验证LoRA模型权重是否正确加载
    print("\n验证LoRA模型权重...")
    verify_and_fix_lora_weights(model)
    
    # 创建数据加载器
    dataloader = SQuADDataLoader(
        data_path=data_path,
        tokenizer=tokenizer,
        block_size=model.config.block_size,
        batch_size=batch_size
    )
    
    # 创建优化器 - 只优化LoRA参数
    trainable_params = []
    for name, param in model.named_parameters():
        if name in model.trainable_param_names:
            trainable_params.append(param)
    
    # 如果没有可训练参数，打印警告
    if not trainable_params:
        print("警告: 没有找到可训练的LoRA参数!")
        trainable_params = list(model.parameters())  # 回退到所有参数
    
    optimizer = jt.optim.AdamW(trainable_params, lr=learning_rate)
    
    # 训练日志
    train_losses = []
    val_losses = []
    
    # 开始训练
    print("开始LoRA微调...")
    print(f"可训练参数数量: {len(trainable_params)}")
    model.train()
    
    # 计算总步数
    steps_per_epoch = max(1, len(dataloader.train_data) // batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    # 训练循环
    step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        epoch_losses = []
        progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for _ in progress_bar:
            # 获取批次数据
            x_batch, y_batch = dataloader.get_batch('train')
            
            # 前向传播
            _, loss = model(x_batch, y_batch)
            
            # 反向传播
            optimizer.backward(loss)
            
            # 在更新参数前检查梯度（如果启用了调试）
            if debug_gradients:
                grad_percent = check_grads(model, optimizer, step, verbose=(step % 10 == 0))
                
                # 如果大部分参数没有梯度，发出警告
                if grad_percent < 50 and step > 5:  # 前几步可能确实没有梯度
                    print(f"警告: 步骤 {step} - 只有 {grad_percent:.1f}% 的LoRA参数有梯度")
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 记录损失
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            train_losses.append(loss_value)
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
            
            # 验证
            if step > 0 and step % eval_every == 0:
                model.eval()
                val_loss = evaluate(model, dataloader)
                val_losses.append(val_loss)
                model.train()
                print(f"Step {step}/{total_steps} - Val Loss: {val_loss:.4f}")
            
            # 保存模型
            if step > 0 and step % save_every == 0:
                save_path = os.path.join(output_dir, f"gpt2_lora_step_{step}.npz")
                model.save_lora_weights(save_path)
                print(f"LoRA权重已保存至: {save_path}")
                
                # 绘制训练曲线
                save_training_curves(train_losses, val_losses, output_dir)
                
            step += 1
                
        # Epoch结束，计算平均损失
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        # 每个epoch结束保存一次
        save_path = os.path.join(output_dir, f"gpt2_lora_epoch_{epoch+1}.npz")
        model.save_lora_weights(save_path)
        
    # 训练结束，保存最终模型
    final_path = os.path.join(output_dir, "gpt2_lora_final.npz")
    model.save_lora_weights(final_path)
    print(f"最终LoRA权重已保存至: {final_path}")
    
    # 保存训练曲线
    save_training_curves(train_losses, val_losses, output_dir)
    
    # 返回模型和训练结果
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": lora_config.target_modules,
            "qv_only": lora_config.qv_only if hasattr(lora_config, 'qv_only') else False
        }
    }
    
    return model, results


# 评估函数
def evaluate(model, dataloader, num_batches=10):
    model.eval()
    losses = []
    
    for _ in range(num_batches):
        x_batch, y_batch = dataloader.get_batch('val')
        with jt.no_grad():
            _, loss = model(x_batch, y_batch)
        losses.append(loss.item())
    
    return sum(losses) / len(losses)


# 保存训练曲线
def save_training_curves(train_losses, val_losses, output_dir):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    
    if val_losses:
        plt.subplot(1, 2, 2)
        eval_steps = list(range(0, len(train_losses), len(train_losses) // len(val_losses)))[:len(val_losses)]
        plt.plot(eval_steps, val_losses)
        plt.title('Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'squad_lora_training_curves.png'))
    plt.close()


# 绘制不同LoRA配置的对比曲线
def plot_lora_comparison(results_list, labels, output_dir):
    """
    绘制不同LoRA配置的训练曲线对比
    
    参数:
    - results_list: 包含多个训练结果的列表
    - labels: 每个训练结果的标签
    - output_dir: 输出目录
    """
    import matplotlib.pyplot as plt
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 绘制训练损失对比
    plt.subplot(2, 1, 1)
    for i, results in enumerate(results_list):
        # 对训练损失进行平滑处理
        window_size = min(10, len(results["train_losses"]) // 10)
        if window_size > 0:
            smoothed_losses = []
            for j in range(len(results["train_losses"]) - window_size + 1):
                smoothed_losses.append(sum(results["train_losses"][j:j+window_size]) / window_size)
        else:
            smoothed_losses = results["train_losses"]
            
        plt.plot(smoothed_losses, label=labels[i])
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制验证损失对比
    plt.subplot(2, 1, 2)
    for i, results in enumerate(results_list):
        if "val_losses" in results and results["val_losses"]:
            plt.plot(results["val_losses"], label=labels[i])
    
    plt.title('Validation Loss Comparison')
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lora_configurations_comparison.png'))
    plt.close()
    
    print(f"LoRA配置对比图已保存至: {os.path.join(output_dir, 'lora_configurations_comparison.png')}")


# 生成回答
def generate_answer(model, tokenizer, context, question, max_length=50):
    model.eval()
    
    # 准备输入
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text)
    
    # 生成回答
    input_tensor = jt.array(np.array(input_ids)).reshape(1, -1)
    
    # 跟踪最近生成的tokens，用于检测重复
    recent_tokens = []
    repetition_penalty = 1.5  # 增加重复惩罚系数
    
    # 增加跟踪部分生成文本的变量，用于检测长句重复
    generated_text_so_far = ""
    repetition_window = 8  # 检查这个窗口大小的重复
    
    # 逐步生成
    for step in range(max_length):
        with jt.no_grad():
            outputs = model(input_tensor)
        
        # 获取下一个token
        next_token_logits = outputs[0, -1, :].numpy()
        
        # 应用温度采样
        temperature = 0.7 if step < 10 else 0.9  # 根据步骤调整温度
        next_token_logits = next_token_logits / temperature
        
        # 应用重复惩罚 - 随着生成步骤增加，惩罚也增加
        if len(recent_tokens) > 0:
            dynamic_penalty = repetition_penalty * (1.0 + step / (max_length * 2))  # 动态惩罚
            for token_id in set(recent_tokens):
                next_token_logits[token_id] /= dynamic_penalty
        
        # 加入nucleus sampling (top-p)和top-k结合
        top_k = 50
        top_p = 0.92
        
        # 先进行top-k过滤
        top_k_indices = np.argsort(-next_token_logits)[:top_k]
        top_k_logits = next_token_logits[top_k_indices]
        
        # 计算概率分布
        top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        # 应用top-p (nucleus sampling)
        sorted_indices = np.argsort(-top_k_probs)
        sorted_probs = top_k_probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # 找到满足top_p的最小集合
        nucleus_indices = sorted_indices[cumulative_probs <= top_p]
        if len(nucleus_indices) == 0:
            nucleus_indices = np.array([sorted_indices[0]])
        
        # 从nucleus中获取真实的token索引和概率
        nucleus_token_indices = top_k_indices[nucleus_indices]
        nucleus_probs = sorted_probs[:len(nucleus_indices)]
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
        
        # 从nucleus中采样
        next_token_id = np.random.choice(nucleus_token_indices, p=nucleus_probs)
        
        # 检查是否生成了过多重复字符
        if len(recent_tokens) >= 3 and recent_tokens[-1] == recent_tokens[-2] == recent_tokens[-3] == next_token_id:
            # 如果连续4个相同token，强制选择不同的token
            # 创建nucleus_token_indices和nucleus_probs的映射关系
            token_to_prob = {token_idx: prob for token_idx, prob in zip(nucleus_token_indices, nucleus_probs)}
            
            # 筛选不等于next_token_id的token
            remaining_indices = [idx for idx in nucleus_token_indices if idx != next_token_id]
            if remaining_indices:
                # 获取对应的概率
                remaining_probs = np.array([token_to_prob[idx] for idx in remaining_indices])
                # 重新归一化概率
                if len(remaining_probs) > 0:
                    remaining_probs = remaining_probs / np.sum(remaining_probs)
                    next_token_id = np.random.choice(remaining_indices, p=remaining_probs)
        
        # 更新最近的tokens
        recent_tokens.append(next_token_id)
        if len(recent_tokens) > 8:  # 增加记忆长度
            recent_tokens.pop(0)
        
        # 检查是否存在长句级别的重复
        next_token_text = tokenizer.decode([next_token_id])
        generated_text_so_far += next_token_text
        
        # 检查生成的文本是否有长句重复
        if len(generated_text_so_far) > repetition_window * 2:
            for window_size in range(3, repetition_window + 1):
                end_text = generated_text_so_far[-window_size:]
                earlier_text = generated_text_so_far[:-window_size]
                
                # 如果末尾文本在前面也出现过，可能是重复
                if end_text in earlier_text and len(end_text.strip()) > 2:
                    # 遇到重复，强制改变生成方向
                    # 创建nucleus_token_indices和nucleus_probs的映射关系
                    token_to_prob = {token_idx: prob for token_idx, prob in zip(nucleus_token_indices, nucleus_probs)}
                    
                    # 筛选不等于next_token_id的token
                    remaining_indices = [idx for idx in nucleus_token_indices if idx != next_token_id]
                    if remaining_indices:
                        # 获取对应的概率
                        remaining_probs = np.array([token_to_prob[idx] for idx in remaining_indices])
                        # 重新归一化概率
                        if len(remaining_probs) > 0:
                            remaining_probs = remaining_probs / np.sum(remaining_probs)
                            next_token_id = np.random.choice(remaining_indices, p=remaining_probs)
                            # 更新生成的文本
                            generated_text_so_far = generated_text_so_far[:-len(next_token_text)] + tokenizer.decode([next_token_id])
                            break
        
        # 添加到序列
        input_ids.append(int(next_token_id))
        input_tensor = jt.array(np.array(input_ids)).reshape(1, -1)
        
        # 如果生成了结束符或达到最大长度，则停止
        if next_token_id == tokenizer.eos_token_id:
            break
    
    # 解码生成的文本
    generated_text = tokenizer.decode(input_ids)
    
    # 提取回答部分
    answer = generated_text.split("Answer:")[-1].strip()
    
    # 额外处理，移除可能的重复字符序列
    import re
    answer = re.sub(r'([^\w\s])\1{2,}', r'\1\1', answer)  # 最多允许连续两个相同的标点符号
    answer = re.sub(r'(.{3,}?)\1{2,}', r'\1', answer)  # 移除重复的短语
    answer = re.sub(r'\s{2,}', ' ', answer)  # 移除多余空格
    answer = re.sub(r'(.)(\1{10,})', r'\1\1\1', answer)  # 限制字符连续重复不超过3次
    
    return answer


# 添加一个函数来验证和修复LoRA模型的权重
def verify_and_fix_lora_weights(model):
    """
    验证LoRA模型的权重是否正确加载，如果发现问题则尝试修复
    
    参数:
    - model: LoRAGPT模型实例
    
    返回:
    - 布尔值，表示是否所有权重都正确
    """
    all_correct = True
    fixed_count = 0
    
    # 检查所有LoRALinear层
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # 检查原始权重是否为零矩阵
            weight_sum = jt.sum(jt.abs(module.weight)).item()
            if weight_sum < 1e-6:  # 如果权重接近零
                print(f"警告: 在 {name} 中发现零权重矩阵，这可能表示预训练权重未正确加载")
                all_correct = False
            
            # 检查lora_A和lora_B是否正确初始化
            a_initialized = jt.sum(jt.abs(module.lora_A.weight)).item() > 0
            b_is_zero = jt.sum(jt.abs(module.lora_B.weight)).item() < 1e-6
            
            if not a_initialized or not b_is_zero:
                print(f"警告: 在 {name} 中LoRA权重初始化不正确")
                all_correct = False
    
    if all_correct:
        print("所有LoRA权重检查通过")
    else:
        print(f"发现 {fixed_count} 个问题并尝试修复")
    
    return all_correct


# 定义单独的LoRA投影层用于Q、K、V
class LoRAQKVProjection(nn.Module):
    def __init__(self, hidden_size, head_size, num_heads, apply_q=True, apply_k=True, apply_v=True, 
                r=8, lora_alpha=16, lora_dropout=0.1, original_weights=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        
        # 是否应用LoRA
        self.apply_q = apply_q
        self.apply_k = apply_k
        self.apply_v = apply_v
        
        # 保存原始投影权重 (hidden_size, 3 * hidden_size)
        if original_weights is not None:
            # 原始权重 (out_features, in_features)形状为(3*hidden_size, hidden_size)
            # 需要转置并拆分成Q、K、V
            w = original_weights
            # 确保权重尺寸正确
            if w.shape[0] == 3 * hidden_size and w.shape[1] == hidden_size:
                # 转置并划分权重
                # 假设权重排列为[Q;K;V]
                w = w.reshape(3, hidden_size, hidden_size)
                self.w_q = w[0].copy()  # (hidden_size, hidden_size)
                self.w_k = w[1].copy()  # (hidden_size, hidden_size)
                self.w_v = w[2].copy()  # (hidden_size, hidden_size)
            else:
                print(f"警告: 无法正确拆分原始权重，形状为{w.shape}，使用随机初始化")
                self.w_q = jt.init.gauss((hidden_size, hidden_size), 'float32', mean=0.0, std=0.02)
                self.w_k = jt.init.gauss((hidden_size, hidden_size), 'float32', mean=0.0, std=0.02)
                self.w_v = jt.init.gauss((hidden_size, hidden_size), 'float32', mean=0.0, std=0.02)
        else:
            # 随机初始化
            self.w_q = jt.init.gauss((hidden_size, hidden_size), 'float32', mean=0.0, std=0.02)
            self.w_k = jt.init.gauss((hidden_size, hidden_size), 'float32', mean=0.0, std=0.02)
            self.w_v = jt.init.gauss((hidden_size, hidden_size), 'float32', mean=0.0, std=0.02)
        
        # 冻结原始权重
        self.w_q.requires_grad = False
        self.w_k.requires_grad = False
        self.w_v.requires_grad = False
        
        # 创建LoRA层 - 为了确保梯度正确传播，我们不再使用条件判断
        # 而是始终创建所有LoRA层，对于不需要的层，我们将缩放设置为0
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling_q = lora_alpha / r if apply_q else 0.0
        self.scaling_k = lora_alpha / r if apply_k else 0.0
        self.scaling_v = lora_alpha / r if apply_v else 0.0
        
        # 为Q创建LoRA层
        self.lora_A_q = nn.Linear(hidden_size, r, bias=False)
        self.lora_B_q = nn.Linear(r, hidden_size, bias=False)
        # 初始化
        self.lora_A_q.weight = jt.init.gauss(self.lora_A_q.weight.shape, 'float32', std=1/r)
        self.lora_B_q.weight = jt.zeros(self.lora_B_q.weight.shape)
        
        # 为K创建LoRA层
        self.lora_A_k = nn.Linear(hidden_size, r, bias=False)
        self.lora_B_k = nn.Linear(r, hidden_size, bias=False)
        # 初始化
        self.lora_A_k.weight = jt.init.gauss(self.lora_A_k.weight.shape, 'float32', std=1/r)
        self.lora_B_k.weight = jt.zeros(self.lora_B_k.weight.shape)
        
        # 为V创建LoRA层
        self.lora_A_v = nn.Linear(hidden_size, r, bias=False)
        self.lora_B_v = nn.Linear(r, hidden_size, bias=False)
        # 初始化
        self.lora_A_v.weight = jt.init.gauss(self.lora_A_v.weight.shape, 'float32', std=1/r)
        self.lora_B_v.weight = jt.zeros(self.lora_B_v.weight.shape)
    
    def execute(self, x):
        # x的形状: (B, T, hidden_size)
        B, T, C = x.shape
        
        # 应用dropout到输入
        dropped_x = self.lora_dropout(x)
        
        # 计算LoRA的调整 - 无论是否应用LoRA，都要计算，以确保梯度传播
        # 对于不需要的层，我们之前设置了缩放为0，所以不会影响结果
        delta_q = self.lora_B_q(self.lora_A_q(dropped_x)) * self.scaling_q
        delta_k = self.lora_B_k(self.lora_A_k(dropped_x)) * self.scaling_k
        delta_v = self.lora_B_v(self.lora_A_v(dropped_x)) * self.scaling_v
        
        # 原始投影
        q = jt.matmul(x, self.w_q.t()) + delta_q
        k = jt.matmul(x, self.w_k.t()) + delta_k
        v = jt.matmul(x, self.w_v.t()) + delta_v
        
        # 调整形状以适应多头注意力
        q = q.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k = k.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        
        return q, k, v


if __name__ == "__main__":
    default_lora_config = LoRAConfig(
        r=4,
        alpha=8,
        dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
        qv_only=False
    )
    
    model1, results1 = train_lora(
        base_model_path='./local_gpt2',
        data_path='./data/processed_squad_100.json',  # 使用100个样本的小数据集进行快速测试
        output_dir='./checkpoints/default_lora',
        batch_size=2,  # 减小批次大小以适应内存限制
        learning_rate=1e-3,  # 增大学习率以加速收敛
        num_epochs=1,  # 减少训练轮次
        save_every=10,  # 更频繁地保存模型
        eval_every=5,  # 更频繁地评估模型
        lora_config=default_lora_config,  # 使用预定义的配置
        debug_gradients=True  # 启用梯度调试
    )

    qv_lora_config = LoRAConfig(
        r=4,
        alpha=8,
        dropout=0.05,
        target_modules=["c_attn"],
        qv_only=True  # 只对Q和V应用LoRA
    )

    model2, results2 = train_lora(
        base_model_path='./local_gpt2',
        data_path='./data/processed_squad_100.json',
        output_dir='./checkpoints/qv_only_lora',
        batch_size=2,
        learning_rate=1e-3,
        num_epochs=1,
        save_every=10,
        eval_every=5,
        max_grad_norm=0.5,
        lora_config=qv_lora_config,  # 使用QV专用配置
        debug_gradients=True  # 启用梯度调试
    )

    ffn_lora_config = LoRAConfig(
        r=4,
        alpha=8,
        dropout=0.05,
        target_modules=["c_fc"],
        qv_only=False
    )

    model3, results3 = train_lora(
        base_model_path='./local_gpt2',
        data_path='./data/processed_squad_100.json',
        output_dir='./checkpoints/ffn_only_lora',
        batch_size=2,
        learning_rate=1e-3,
        num_epochs=1,
        save_every=10,
        eval_every=5,
        max_grad_norm=0.5,
        lora_config=ffn_lora_config,  # 使用FFN专用配置
        debug_gradients=True  # 启用梯度调试
    )

    # 绘制不同LoRA配置的对比曲线
    os.makedirs('./checkpoints/comparison', exist_ok=True)
    plot_lora_comparison(
        [results1, results2, results3],
        ["Default (c_attn, c_proj, c_fc)", "QV Only", "FFN Only (c_fc)"],
        './checkpoints/comparison'
    )

    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 测试生成回答
    context = "Jittor is a high-performance deep learning framework based on JIT compilation."
    question = "What is Jittor?"

    # 使用默认LoRA配置的模型生成回答
    answer1 = generate_answer(model1, tokenizer, context, question)
    print(f"\n默认LoRA配置 (c_attn, c_proj, c_fc):")
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Generated Answer: {answer1}")

    # 使用仅注意力LoRA配置的模型生成回答
    answer2 = generate_answer(model2, tokenizer, context, question)
    print(f"\n仅注意力LoRA配置 (c_attn):")
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Generated Answer: {answer2}")

    # 使用仅前馈网络LoRA配置的模型生成回答
    answer3 = generate_answer(model3, tokenizer, context, question)
    print(f"\n仅前馈网络LoRA配置 (c_fc):")
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Generated Answer: {answer3}")
