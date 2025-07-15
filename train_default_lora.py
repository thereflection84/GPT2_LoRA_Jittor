import os
import numpy as np
import jittor as jt
from tqdm import tqdm
from transformers import GPT2Tokenizer

# 显式启用GPU
jt.flags.use_cuda = 1

# 导入自定义模块
from GPT2_jittor import GPT
from LoRA import LoRAConfig, check_grads, verify_and_fix_lora_weights
from lora_models import LoRAGPT
from model_utils import SQuADDataLoader, evaluate
from plot_loss import save_training_curves

# 训练函数
def train_default_lora(
    base_model_path='./local_gpt2',
    data_path='./data/processed_squad_100.json',
    output_dir='./checkpoints/default_lora',
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    batch_size=1,  # 使用最小batch_size以减少内存使用
    learning_rate=5e-4,
    num_epochs=3,
    save_every=100,
    eval_every=50,
    target_modules=["c_attn", "c_proj", "c_fc"],
    qv_only=False
):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 加载基础模型
    print(f"加载基础模型: {base_model_path}")
    base_model = GPT.from_pretrained(base_model_path)
    
    # 创建LoRA配置
    lora_config = LoRAConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=target_modules,
        qv_only=qv_only
    )
    
    # 创建LoRA模型
    print("创建LoRA模型...")
    model = LoRAGPT(base_model, lora_config)
    model.print_trainable_parameters()
    
    # 验证LoRA模型权重是否正确加载
    print("\n验证LoRA模型权重...")
    verify_and_fix_lora_weights(model)
    
    # 创建数据加载器
    print(f"加载训练数据: {data_path}")
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
            
            # 在更新参数前检查梯度
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
                
                # 保存当前的训练损失
                save_losses(train_losses, val_losses, output_dir)
                
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
    
    # 保存最终的训练损失
    save_losses(train_losses, val_losses, output_dir)
    
    # 绘制训练曲线
    save_training_curves(train_losses, val_losses, output_dir)
    
    return model, train_losses, val_losses


# 保存损失到txt文件
def save_losses(train_losses, val_losses, output_dir):
    """保存训练和验证损失到txt文件"""
    # 保存训练损失
    train_loss_path = os.path.join(output_dir, "train_loss.txt")
    with open(train_loss_path, "w") as f:
        for loss in train_losses:
            f.write(f"{loss}\n")
    
    # 保存验证损失
    if val_losses:
        val_loss_path = os.path.join(output_dir, "val_loss.txt")
        with open(val_loss_path, "w") as f:
            for loss in val_losses:
                f.write(f"{loss}\n")
    
    print(f"训练损失已保存至: {train_loss_path}")
    if val_losses:
        print(f"验证损失已保存至: {val_loss_path}")


if __name__ == "__main__":
    # 训练默认LoRA模型
    model, train_losses, val_losses = train_default_lora(
        base_model_path='./local_gpt2',
        data_path='./data/processed_squad_100.json',
        output_dir='./checkpoints/default_lora',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        batch_size=1,  # 使用最小batch_size以减少内存使用
        learning_rate=5e-4,
        num_epochs=3,
        save_every=100,
        eval_every=50,
        target_modules=["c_attn", "c_proj", "c_fc"],
        qv_only=False
    ) 
