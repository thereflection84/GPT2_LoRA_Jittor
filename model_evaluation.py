import os
import json
import numpy as np
import re
import time
from tqdm import tqdm
from transformers import GPT2Tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from collections import defaultdict
import jittor as jt
from GPT2_jittor import GPT
from GPT2_LoRA import LoRAGPT, LoRAConfig, generate_answer
jt.flags.use_cuda = 1
JITTOR_AVAILABLE = True

# 配置matplotlib支持中文
def configure_matplotlib_chinese():
    """配置matplotlib以支持中文显示"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 尝试查找系统中的中文字体
        chinese_fonts = []
        
        # Windows常见中文字体
        windows_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
        # macOS常见中文字体
        macos_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'STKaiti', 'STSong', 'STFangsong']
        # Linux常见中文字体
        linux_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC']
        
        all_fonts = windows_fonts + macos_fonts + linux_fonts
        
        # 获取系统字体列表
        font_paths = fm.findSystemFonts()
        font_objects = [fm.FontProperties(fname=font_path) for font_path in font_paths]
        font_names = [font.get_name() for font in font_objects]
        
        # 查找可用的中文字体
        for font in all_fonts:
            if font in font_names:
                chinese_fonts.append(font)
                break
        
        # 如果找到中文字体，配置matplotlib
        if chinese_fonts:
            plt.rcParams['font.family'] = chinese_fonts[0]
            print(f"已配置matplotlib使用中文字体: {chinese_fonts[0]}")
        else:
            # 如果没找到，尝试使用Matplotlib内置的中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print("未找到系统中文字体，使用matplotlib内置字体")
    except Exception as e:
        print(f"配置matplotlib中文字体时出错: {e}")
        print("继续使用默认字体")

# 自动下载NLTK资源
def download_nltk_resources():
    """下载评估所需的NLTK资源"""
    try:
        print("检查并下载NLTK资源...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK资源下载完成")
    except Exception as e:
        print(f"下载NLTK资源时出错: {e}")
        print("请手动运行以下命令下载资源:")
        print("import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')")

# 在脚本开始时下载资源
download_nltk_resources()
configure_matplotlib_chinese()

# 评估配置
class EvaluationConfig:
    def __init__(self):
        self.base_model_path = './local_gpt2'
        self.lora_weights_path = './checkpoints/default_lora/gpt2_lora_final.npz'
        self.output_dir = './evaluation_results'
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.target_modules = ["c_attn", "c_proj", "c_fc"]
        self.max_samples = 100  # 最大评估样本数
        self.max_length = 50    # 生成答案的最大长度

# 加载评估数据
def load_evaluation_data(data_path, max_samples=100):
    """加载评估数据集"""
    # 如果是SQuAD格式的数据
    if data_path.endswith('.json'):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            eval_samples = []
            if 'data' in data:  # 原始SQuAD格式
                for article in data['data']:
                    for paragraph in article['paragraphs']:
                        context = paragraph['context']
                        for qa in paragraph['qas']:
                            if len(eval_samples) >= max_samples:
                                break
                            question = qa['question']
                            if qa['answers']:
                                answer = qa['answers'][0]['text']
                                eval_samples.append({
                                    'context': context,
                                    'question': question,
                                    'reference_answer': answer
                                })
            elif 'train' in data or 'val' in data:  # 处理过的格式
                samples = data.get('val', data.get('train', []))
                for i, sample in enumerate(samples):
                    if i >= max_samples:
                        break
                    if 'tokens' in sample:
                        # 需要解码tokens
                        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                        text = tokenizer.decode(sample['tokens'])
                        # 尝试提取context, question和answer
                        match = re.search(r'Context: (.*?)\nQuestion: (.*?)\nAnswer: (.*?)(?:\n|$)', text, re.DOTALL)
                        if match:
                            context, question, answer = match.groups()
                            eval_samples.append({
                                'context': context.strip(),
                                'question': question.strip(),
                                'reference_answer': answer.strip()
                            })
            
            print(f"从{data_path}加载了{len(eval_samples)}个评估样本")
            return eval_samples
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return []
    else:
        print(f"不支持的数据格式: {data_path}")
        return []

# 计算评估指标
def calculate_metrics(reference, hypothesis):
    """计算各种NLP评估指标"""
    metrics = {}
    
    # 准备数据
    if not hypothesis or not reference:
        return {
            'bleu-1': 0.0,
            'bleu-2': 0.0,
            'bleu-4': 0.0,
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
            'meteor': 0.0
        }
    
    # 分词
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    
    # BLEU分数
    smoothie = SmoothingFunction().method1
    try:
        metrics['bleu-1'] = sentence_bleu([reference_tokens], hypothesis_tokens, 
                                         weights=(1, 0, 0, 0), 
                                         smoothing_function=smoothie)
        metrics['bleu-2'] = sentence_bleu([reference_tokens], hypothesis_tokens, 
                                         weights=(0.5, 0.5, 0, 0), 
                                         smoothing_function=smoothie)
        metrics['bleu-4'] = sentence_bleu([reference_tokens], hypothesis_tokens, 
                                         weights=(0.25, 0.25, 0.25, 0.25), 
                                         smoothing_function=smoothie)
    except Exception as e:
        print(f"计算BLEU分数时出错: {e}")
        metrics['bleu-1'] = metrics['bleu-2'] = metrics['bleu-4'] = 0.0
    
    # ROUGE分数
    try:
        rouge = Rouge()
        rouge_scores = rouge.get_scores(hypothesis, reference)
        metrics['rouge-1'] = rouge_scores[0]['rouge-1']['f']
        metrics['rouge-2'] = rouge_scores[0]['rouge-2']['f']
        metrics['rouge-l'] = rouge_scores[0]['rouge-l']['f']
    except Exception as e:
        print(f"计算ROUGE分数时出错: {e}")
        metrics['rouge-1'] = metrics['rouge-2'] = metrics['rouge-l'] = 0.0
    
    # METEOR分数
    try:
        metrics['meteor'] = meteor_score([reference_tokens], hypothesis_tokens)
    except Exception as e:
        print(f"计算METEOR分数时出错: {e}")
        print("尝试再次下载wordnet资源...")
        try:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            # 重试计算METEOR
            metrics['meteor'] = meteor_score([reference_tokens], hypothesis_tokens)
        except Exception as e2:
            print(f"重试计算METEOR分数时出错: {e2}")
            metrics['meteor'] = 0.0
    
    return metrics

# 评估模型
def evaluate_models(config):
    """评估预训练模型和LoRA模型的性能"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 加载评估数据
    eval_data = load_evaluation_data('./data/dev-v1.1.json', config.max_samples)
    if not eval_data:
        print("错误: 无法加载评估数据")
        return
    
    # 初始化模型（如果Jittor可用）
    base_model = None
    lora_model = None
    
    if JITTOR_AVAILABLE:
        try:
            print(f"加载基础模型: {config.base_model_path}")
            base_model = GPT.from_pretrained(config.base_model_path)
            
            print("创建LoRA模型...")
            lora_config = LoRAConfig(
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.target_modules
            )
            lora_model = LoRAGPT(base_model, lora_config)
            
            print(f"加载LoRA权重: {config.lora_weights_path}")
            success = lora_model.load_lora_weights(config.lora_weights_path)
            if not success:
                print("警告: 加载LoRA权重失败，将使用模拟评估")
                lora_model = None
        except Exception as e:
            print(f"加载模型时出错: {e}")
            base_model = lora_model = None
    
    # 评估结果
    results = {
        'base_model': {'answers': [], 'metrics': defaultdict(float)},
        'lora_model': {'answers': [], 'metrics': defaultdict(float)}
    }
    
    # 评估每个样本
    print("\n开始评估...")
    for i, sample in enumerate(tqdm(eval_data)):
        context = sample['context']
        question = sample['question']
        reference = sample['reference_answer']
        
        # 基础模型生成
        if base_model and JITTOR_AVAILABLE:
            base_answer = generate_answer(base_model, tokenizer, context, question, max_length=config.max_length)
        
        # LoRA模型生成
        if lora_model and JITTOR_AVAILABLE:
            lora_answer = generate_answer(lora_model, tokenizer, context, question, max_length=config.max_length)
        
        # 计算指标
        base_metrics = calculate_metrics(reference, base_answer)
        lora_metrics = calculate_metrics(reference, lora_answer)
        
        # 保存结果
        results['base_model']['answers'].append({
            'context': context,
            'question': question,
            'reference': reference,
            'generated': base_answer,
            'metrics': base_metrics
        })
        
        results['lora_model']['answers'].append({
            'context': context,
            'question': question,
            'reference': reference,
            'generated': lora_answer,
            'metrics': lora_metrics
        })
        
        # 累加指标
        for metric, value in base_metrics.items():
            results['base_model']['metrics'][metric] += value
        
        for metric, value in lora_metrics.items():
            results['lora_model']['metrics'][metric] += value
    
    # 计算平均指标
    sample_count = len(eval_data)
    for model_type in ['base_model', 'lora_model']:
        for metric in results[model_type]['metrics']:
            results[model_type]['metrics'][metric] /= sample_count
    
    # 保存详细结果
    detailed_results_file = os.path.join(config.output_dir, "detailed_evaluation_results.json")
    with open(detailed_results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 创建摘要结果
    summary = {
        'sample_count': sample_count,
        'base_model': {
            'metrics': dict(results['base_model']['metrics'])
        },
        'lora_model': {
            'metrics': dict(results['lora_model']['metrics'])
        },
        'improvement': {}
    }
    
    # 计算改进百分比
    for metric in summary['base_model']['metrics']:
        base_value = summary['base_model']['metrics'][metric]
        lora_value = summary['lora_model']['metrics'][metric]
        if base_value > 0:
            improvement = ((lora_value - base_value) / base_value) * 100
        else:
            improvement = float('inf') if lora_value > 0 else 0
        summary['improvement'][metric] = improvement
    
    # 保存摘要结果
    summary_file = os.path.join(config.output_dir, "evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n评估完成!")
    print(f"详细结果已保存至: {detailed_results_file}")
    print(f"摘要结果已保存至: {summary_file}")
    
    print("\n模型性能对比:")
    print("-" * 60)
    print(f"{'指标':<10} {'基础模型':<15} {'LoRA模型':<15} {'改进(%)':<10}")
    print("-" * 60)
    
    for metric in sorted(summary['base_model']['metrics'].keys()):
        base_value = summary['base_model']['metrics'][metric]
        lora_value = summary['lora_model']['metrics'][metric]
        improvement = summary['improvement'][metric]
        print(f"{metric:<10} {base_value:<15.4f} {lora_value:<15.4f} {improvement:<10.2f}")
    
    print("-" * 60)
    
    # 生成可视化报告
    generate_visualization(summary, config.output_dir)
    
    return summary

# 生成可视化报告
def generate_visualization(summary, output_dir):
    """生成评估结果的可视化图表"""
    try:
        import matplotlib.pyplot as plt
        
        # 准备数据
        metrics = list(summary['base_model']['metrics'].keys())
        base_values = [summary['base_model']['metrics'][m] for m in metrics]
        lora_values = [summary['lora_model']['metrics'][m] for m in metrics]
        
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
        
        print(f"\n性能对比图表已保存至: {chart_path}")
        
        # 创建改进百分比图
        plt.figure(figsize=(12, 8))
        improvements = [summary['improvement'][m] for m in metrics]
        
        plt.bar(x, improvements, width, color='green')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        plt.xlabel('评估指标')
        plt.ylabel('改进百分比 (%)')
        plt.title('LoRA模型相对于基础模型的改进百分比')
        plt.xticks(x, metrics)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, v in enumerate(improvements):
            plt.text(i, v + (5 if v >= 0 else -10), 
                     f"{v:.1f}%", 
                     ha='center', va='bottom' if v >= 0 else 'top')
        
        # 保存图表
        improvement_chart_path = os.path.join(output_dir, 'improvement_percentage_chart.png')
        plt.tight_layout()
        plt.savefig(improvement_chart_path)
        plt.close()
        
        print(f"改进百分比图表已保存至: {improvement_chart_path}")
        
    except ImportError:
        print("警告: 无法导入matplotlib，跳过可视化生成")
    except Exception as e:
        print(f"生成可视化时出错: {e}")

if __name__ == "__main__":
    # 运行评估
    config = EvaluationConfig()
    evaluate_models(config) 