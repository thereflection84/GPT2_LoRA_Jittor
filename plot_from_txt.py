import os
import numpy as np
import matplotlib.pyplot as plt

def plot_from_txt(train_loss_file, val_loss_file=None, output_dir=None, title="LoRA训练曲线"):
    """
    从txt文件读取损失值并绘制训练曲线
    
    参数:
    - train_loss_file: 训练损失文件路径
    - val_loss_file: 验证损失文件路径（可选）
    - output_dir: 输出目录（可选）
    - title: 图表标题
    """
    # 读取训练损失
    train_losses = []
    with open(train_loss_file, 'r') as f:
        for line in f:
            try:
                loss = float(line.strip())
                train_losses.append(loss)
            except ValueError:
                print(f"警告: 无法解析行 '{line.strip()}' 为浮点数，已跳过")
    
    # 读取验证损失（如果有）
    val_losses = []
    if val_loss_file and os.path.exists(val_loss_file):
        with open(val_loss_file, 'r') as f:
            for line in f:
                try:
                    loss = float(line.strip())
                    val_losses.append(loss)
                except ValueError:
                    print(f"警告: 无法解析行 '{line.strip()}' 为浮点数，已跳过")
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    plt.subplot(1, 2 if val_losses else 1, 1)
    plt.plot(train_losses)
    plt.title('训练损失')
    plt.xlabel('步骤')
    plt.ylabel('损失')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 如果有验证损失，绘制验证损失
    if val_losses:
        plt.subplot(1, 2, 2)
        # 计算评估步骤（假设每N步评估一次）
        eval_interval = len(train_losses) // len(val_losses)
        eval_steps = list(range(0, len(train_losses), eval_interval))[:len(val_losses)]
        plt.plot(eval_steps, val_losses)
        plt.title('验证损失')
        plt.xlabel('步骤')
        plt.ylabel('损失')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置总标题
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间
    
    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'lora_training_curves.png')
        plt.savefig(output_file)
        print(f"训练曲线已保存至: {output_file}")
    
    # 显示图表
    plt.show()


def plot_smoothed_loss(train_loss_file, window_size=10, output_dir=None, title="LoRA训练曲线（平滑）"):
    """
    从txt文件读取损失值，应用平滑处理并绘制训练曲线
    
    参数:
    - train_loss_file: 训练损失文件路径
    - window_size: 平滑窗口大小
    - output_dir: 输出目录（可选）
    - title: 图表标题
    """
    # 读取训练损失
    train_losses = []
    with open(train_loss_file, 'r') as f:
        for line in f:
            try:
                loss = float(line.strip())
                train_losses.append(loss)
            except ValueError:
                print(f"警告: 无法解析行 '{line.strip()}' 为浮点数，已跳过")
    
    # 应用移动平均平滑
    smoothed_losses = []
    for i in range(len(train_losses)):
        window_start = max(0, i - window_size + 1)
        window = train_losses[window_start:i+1]
        smoothed_losses.append(sum(window) / len(window))
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制原始和平滑后的训练损失
    plt.plot(train_losses, alpha=0.3, label='原始损失')
    plt.plot(smoothed_losses, linewidth=2, label=f'平滑损失 (窗口={window_size})')
    
    plt.title(title)
    plt.xlabel('步骤')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'smoothed_training_curve.png')
        plt.savefig(output_file)
        print(f"平滑训练曲线已保存至: {output_file}")
    
    # 显示图表
    plt.show()


if __name__ == "__main__":
    # 默认路径
    train_loss_file = "./checkpoints/default_lora/train_loss.txt"
    val_loss_file = "./checkpoints/default_lora/val_loss.txt"
    output_dir = "./checkpoints/default_lora"
    
    # 绘制训练曲线
    if os.path.exists(train_loss_file):
        plot_from_txt(train_loss_file, val_loss_file, output_dir)
        
        # 绘制平滑后的训练曲线
        plot_smoothed_loss(train_loss_file, window_size=20, output_dir=output_dir)
    else:
        print(f"错误: 训练损失文件 {train_loss_file} 不存在") 
