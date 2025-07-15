import os
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.font_manager as fm
import platform

# 配置matplotlib支持中文显示
def configure_matplotlib_chinese():
    """配置matplotlib以支持中文显示"""
    system = platform.system()
    
    # 根据不同操作系统选择合适的中文字体
    if system == 'Windows':
        # Windows系统常用中文字体
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    elif system == 'Darwin':  # macOS
        # macOS系统常用中文字体
        chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'STKaiti', 'STSong', 'STFangsong']
    else:  # Linux或其他系统
        # Linux系统常用中文字体
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC']
    
    # 查找可用的中文字体
    available_font = None
    font_paths = fm.findSystemFonts()
    font_objects = [fm.FontProperties(fname=font_path) for font_path in font_paths]
    font_names = [font.get_name() for font in font_objects]
    
    for font in chinese_fonts:
        if font in font_names:
            available_font = font
            break
    
    # 如果找到中文字体，配置matplotlib
    if available_font:
        plt.rcParams['font.family'] = available_font
        print(f"已配置matplotlib使用中文字体: {available_font}")
    else:
        # 如果没找到，尝试使用Matplotlib内置的支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("未找到系统中文字体，使用matplotlib内置字体")

# 在脚本开始时配置中文字体
configure_matplotlib_chinese()

# 保存训练曲线
def save_training_curves(train_losses, val_losses, output_dir):
    """
    保存训练和验证损失曲线
    
    参数:
    - train_losses: 训练损失列表
    - val_losses: 验证损失列表
    - output_dir: 输出目录
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('训练损失')
    plt.xlabel('步骤')
    plt.ylabel('损失')
    
    if val_losses:
        plt.subplot(1, 2, 2)
        eval_steps = list(range(0, len(train_losses), len(train_losses) // len(val_losses)))[:len(val_losses)]
        plt.plot(eval_steps, val_losses)
        plt.title('验证损失')
        plt.xlabel('步骤')
        plt.ylabel('损失')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lora_training_curves.png'))
    plt.close()
    print(f"训练曲线已保存至: {os.path.join(output_dir, 'lora_training_curves.png')}")


# 绘制不同LoRA配置的对比曲线
def plot_lora_comparison(results_list, labels, output_dir):
    """
    绘制不同LoRA配置的训练曲线对比
    
    参数:
    - results_list: 包含多个训练结果的列表
    - labels: 每个训练结果的标签
    - output_dir: 输出目录
    """
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
    
    plt.title('训练损失对比')
    plt.xlabel('步骤')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制验证损失对比
    plt.subplot(2, 1, 2)
    for i, results in enumerate(results_list):
        if "val_losses" in results and results["val_losses"]:
            plt.plot(results["val_losses"], label=labels[i])
    
    plt.title('验证损失对比')
    plt.xlabel('评估步骤')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lora_configurations_comparison.png'))
    plt.close()
    
    print(f"LoRA配置对比图已保存至: {os.path.join(output_dir, 'lora_configurations_comparison.png')}")


# 保存实验结果到JSON文件
def save_experiment_results(results_list, labels, configs, output_dir):
    """
    保存实验结果到JSON文件
    
    参数:
    - results_list: 包含多个训练结果的列表
    - labels: 每个训练结果的标签
    - configs: 每个实验的配置
    - output_dir: 输出目录
    """
    experiment_results = []
    
    for i, (results, label, config) in enumerate(zip(results_list, labels, configs)):
        # 计算平均训练损失和最终训练损失
        avg_train_loss = sum(results["train_losses"]) / len(results["train_losses"]) if results["train_losses"] else 0
        final_train_loss = results["train_losses"][-1] if results["train_losses"] else 0
        
        # 计算平均验证损失和最终验证损失
        avg_val_loss = sum(results["val_losses"]) / len(results["val_losses"]) if results["val_losses"] else 0
        final_val_loss = results["val_losses"][-1] if results["val_losses"] else 0
        
        # 创建结果字典
        result_dict = {
            "experiment": label,
            "config": config,
            "metrics": {
                "avg_train_loss": avg_train_loss,
                "final_train_loss": final_train_loss,
                "avg_val_loss": avg_val_loss,
                "final_val_loss": final_val_loss
            }
        }
        
        experiment_results.append(result_dict)
    
    # 保存到JSON文件
    output_file = os.path.join(output_dir, "lora_experiment_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, ensure_ascii=False, indent=2)
    
    print(f"实验结果已保存至: {output_file}")


# 绘制模型评估结果对比
def plot_evaluation_comparison(base_metrics, lora_metrics, output_dir):
    """
    绘制基础模型和LoRA模型的评估指标对比
    
    参数:
    - base_metrics: 基础模型的评估指标
    - lora_metrics: LoRA模型的评估指标
    - output_dir: 输出目录
    """
    metrics = list(base_metrics.keys())
    base_values = [base_metrics[m] for m in metrics]
    lora_values = [lora_metrics[m] for m in metrics]
    
    # 创建柱状图
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, base_values, width, label='基础模型')
    plt.bar(x + width/2, lora_values, width, label='LoRA模型')
    
    plt.xlabel('评估指标')
    plt.ylabel('分数')
    plt.title('基础模型 vs LoRA模型 性能对比')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    chart_path = os.path.join(output_dir, 'model_comparison_chart.png')
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    
    print(f"性能对比图表已保存至: {chart_path}")
    
    # 计算改进百分比
    improvements = {}
    for metric in metrics:
        base_value = base_metrics[metric]
        lora_value = lora_metrics[metric]
        if base_value > 0:
            improvement = ((lora_value - base_value) / base_value) * 100
        else:
            improvement = float('inf') if lora_value > 0 else 0
        improvements[metric] = improvement
    
    # 绘制改进百分比图
    plt.figure(figsize=(12, 8))
    improvement_values = [improvements[m] for m in metrics]
    
    plt.bar(x, improvement_values, width, color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xlabel('评估指标')
    plt.ylabel('改进百分比 (%)')
    plt.title('LoRA模型相对于基础模型的改进百分比')
    plt.xticks(x, metrics)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(improvement_values):
        plt.text(i, v + (5 if v >= 0 else -10), 
                 f"{v:.1f}%", 
                 ha='center', va='bottom' if v >= 0 else 'top')
    
    # 保存图表
    improvement_chart_path = os.path.join(output_dir, 'improvement_percentage_chart.png')
    plt.tight_layout()
    plt.savefig(improvement_chart_path)
    plt.close()
    
    print(f"改进百分比图表已保存至: {improvement_chart_path}")
    
    return improvements 
