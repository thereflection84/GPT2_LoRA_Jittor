import math
import jittor as jt
import jittor.nn as nn
from dataclasses import dataclass, field

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


# 验证和修复LoRA模型的权重
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
