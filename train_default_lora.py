import os
import numpy as np
import jittor as jt
from tqdm import tqdm
import time
import datetime
import json
import logging
import psutil
from transformers import GPT2Tokenizer
import platform

# 显式启用GPU
jt.flags.use_cuda = 1

# 导入自定义模块
from GPT2_jittor import GPT
from LoRA import LoRAConfig, check_grads, verify_and_fix_lora_weights
from lora_models import LoRAGPT
from model_utils import SQuADDataLoader, evaluate, generate_answer
from plot_loss import save_training_curves

# 设置日志记录器
def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # 同时输出到控制台
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

# 获取系统性能信息
def get_performance_metrics():
    """获取系统性能指标"""
    metrics = {}
    
    # CPU使用率
    metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    
    # 内存使用情况
    memory = psutil.virtual_memory()
    metrics['memory_used_gb'] = memory.used / (1024 ** 3)
    metrics['memory_percent'] = memory.percent
    
    # GPU信息（如果可用）
    try:
        # 使用jittor的CUDA标志检查GPU是否可用
        metrics['gpu_available'] = bool(jt.flags.use_cuda)
        
        # 由于jittor没有直接的API获取GPU内存使用情况
        # 这里只记录GPU是否可用，详细信息通过display_memory_info打印到控制台
        if jt.flags.use_cuda:
            # 打印内存信息到控制台，但不直接获取数值
            jt.display_memory_info()
    except Exception as e:
        metrics['gpu_error'] = str(e)
    
    return metrics

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
    qv_only=False,
    log_performance_every=10  # 每隔多少步记录一次性能指标
):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志记录器
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建实验日志和性能日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    performance_log_file = os.path.join(log_dir, f"performance_{timestamp}.log")
    
    experiment_logger = setup_logger('experiment_logger', experiment_log_file)
    performance_logger = setup_logger('performance_logger', performance_log_file)
    
    # 记录实验配置
    config = {
        'base_model_path': base_model_path,
        'data_path': data_path,
        'output_dir': output_dir,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'save_every': save_every,
        'eval_every': eval_every,
        'target_modules': target_modules,
        'qv_only': qv_only
    }
    
    experiment_logger.info(f"实验配置: {json.dumps(config, ensure_ascii=False, indent=2)}")
    
    # 记录系统信息
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3)
    }
    
    experiment_logger.info(f"系统信息: {json.dumps(system_info, ensure_ascii=False, indent=2)}")
    
    # 加载分词器
    experiment_logger.info("加载分词器...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 加载基础模型
    experiment_logger.info(f"加载基础模型: {base_model_path}")
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
    experiment_logger.info("创建LoRA模型...")
    model = LoRAGPT(base_model, lora_config)
    model.print_trainable_parameters()
    
    # 记录模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for name, p in model.named_parameters() if name in model.trainable_param_names)
    
    experiment_logger.info(f"模型总参数数量: {total_params:,}")
    experiment_logger.info(f"可训练参数数量: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)")
    
    # 验证LoRA模型权重是否正确加载
    experiment_logger.info("\n验证LoRA模型权重...")
    verify_and_fix_lora_weights(model)
    
    # 创建数据加载器
    experiment_logger.info(f"加载训练数据: {data_path}")
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
        experiment_logger.warning("警告: 没有找到可训练的LoRA参数!")
        trainable_params = list(model.parameters())  # 回退到所有参数
    
    optimizer = jt.optim.AdamW(trainable_params, lr=learning_rate)
    
    # 训练日志
    train_losses = []
    val_losses = []
    
    # 开始训练
    experiment_logger.info("开始LoRA微调...")
    experiment_logger.info(f"可训练参数数量: {len(trainable_params)}")
    model.train()
    
    # 计算总步数
    steps_per_epoch = max(1, len(dataloader.train_data) // batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    experiment_logger.info(f"每个epoch的步数: {steps_per_epoch}")
    experiment_logger.info(f"总训练步数: {total_steps}")
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    # 训练循环
    step = 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        experiment_logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        epoch_losses = []
        progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for _ in progress_bar:
            step_start_time = time.time()
            
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
                experiment_logger.warning(f"警告: 步骤 {step} - 只有 {grad_percent:.1f}% 的LoRA参数有梯度")
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 记录损失
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            train_losses.append(loss_value)
            
            # 计算步骤耗时
            step_time = time.time() - step_start_time
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
            
            # 记录训练日志
            experiment_logger.info(f"步骤 {step}/{total_steps} - 训练损失: {loss_value:.6f} - 步骤耗时: {step_time:.2f}秒")
            
            # 记录性能指标
            if step % log_performance_every == 0:
                metrics = get_performance_metrics()
                metrics['step'] = step
                metrics['loss'] = loss_value
                metrics['step_time'] = step_time
                performance_logger.info(json.dumps(metrics))
            
            # 验证
            if step > 0 and step % eval_every == 0:
                eval_start_time = time.time()
                model.eval()
                val_loss = evaluate(model, dataloader)
                val_losses.append(val_loss)
                model.train()
                eval_time = time.time() - eval_start_time
                
                experiment_logger.info(f"步骤 {step}/{total_steps} - 验证损失: {val_loss:.6f} - 验证耗时: {eval_time:.2f}秒")
            
            # 保存模型
            if step > 0 and step % save_every == 0:
                save_start_time = time.time()
                save_path = os.path.join(output_dir, f"gpt2_lora_step_{step}.npz")
                model.save_lora_weights(save_path)
                save_time = time.time() - save_start_time
                
                experiment_logger.info(f"步骤 {step}/{total_steps} - 模型已保存至: {save_path} - 保存耗时: {save_time:.2f}秒")
                
                # 保存当前的训练损失
                save_losses(train_losses, val_losses, output_dir)
                
            step += 1
                
        # Epoch结束，计算平均损失
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        epoch_time = time.time() - epoch_start_time
        
        experiment_logger.info(f"Epoch {epoch+1} 完成 - 平均损失: {avg_loss:.6f} - 耗时: {epoch_time:.2f}秒")
        
        # 每个epoch结束保存一次
        save_path = os.path.join(output_dir, f"gpt2_lora_epoch_{epoch+1}.npz")
        model.save_lora_weights(save_path)
        experiment_logger.info(f"Epoch {epoch+1} 模型已保存至: {save_path}")
        
    # 训练结束，保存最终模型
    final_path = os.path.join(output_dir, "gpt2_lora_final.npz")
    model.save_lora_weights(final_path)
    
    # 计算总训练时间
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    experiment_logger.info(f"训练完成! 总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    experiment_logger.info(f"最终LoRA权重已保存至: {final_path}")
    
    # 保存最终的训练损失
    save_losses(train_losses, val_losses, output_dir)
    
    # 绘制训练曲线
    save_training_curves(train_losses, val_losses, output_dir)
    experiment_logger.info(f"训练曲线已保存至: {os.path.join(output_dir, 'lora_training_curves.png')}")
    
    # 保存训练摘要
    summary = {
        'total_steps': step,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'avg_train_loss': sum(train_losses) / len(train_losses) if train_losses else None,
        'avg_val_loss': sum(val_losses) / len(val_losses) if val_losses else None,
        'total_training_time_seconds': total_training_time,
        'training_time_formatted': f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
    }
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    experiment_logger.info(f"训练摘要已保存至: {summary_path}")
    
    return model, train_losses, val_losses, experiment_logger


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


# 测试模型
def test_model(model, tokenizer, test_samples=None, logger=None):
    """
    测试模型在一些示例上的表现
    
    参数:
    - model: 训练好的模型
    - tokenizer: 分词器
    - test_samples: 测试样本列表，如果为None则使用默认样本
    - logger: 日志记录器，如果为None则使用print输出
    
    返回:
    - results: 测试结果列表
    """
    # 日志记录函数
    log = logger.info if logger else print
    
    # 如果没有提供测试样本，使用默认样本
    if test_samples is None:
        test_samples = [
            {
                "context": "Jittor is a high-performance deep learning framework based on Just-in-time compilation technology, developed by the Computer Science Department at Tsinghua University. The main features of Jittor include high usability, superior performance, and support for automatic differentiation and just-in-time compilation techniques.",
                "question": "What is Jittor?"
            },
            {
                "context": "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that works by freezing the pre-trained model weights and injecting trainable low-rank matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.",
                "question": "What is the main purpose of LoRA?"
            },
            {
                "context": "The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset released by Stanford University, consisting of over 100,000 question-answer pairs. In the SQuAD task, given a text passage and a question, the model needs to find the answer from the text.",
                "question": "How many question-answer pairs does SQuAD contain?"
            }
        ]
    
    log("\n开始模型测试...")
    log("=" * 50)
    
    results = []
    model.eval()
    
    for i, sample in enumerate(test_samples):
        context = sample["context"]
        question = sample["question"]
        
        log(f"\n测试样本 {i+1}:")
        log(f"上下文: {context}")
        log(f"问题: {question}")
        
        # 记录生成开始时间
        gen_start_time = time.time()
        
        # 生成答案
        answer = generate_answer(model, tokenizer, context, question, max_length=50)
        
        # 计算生成时间
        gen_time = time.time() - gen_start_time
        
        log(f"模型回答: {answer}")
        log(f"生成耗时: {gen_time:.2f}秒")
        log("-" * 50)
        
        results.append({
            "context": context,
            "question": question,
            "answer": answer,
            "generation_time": gen_time
        })
    
    # 保存测试结果
    return results


# 加载模型并测试
def load_and_test_model(
    base_model_path='./local_gpt2',
    lora_weights_path='./checkpoints/default_lora/gpt2_lora_final.npz',
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj", "c_fc"],
    qv_only=False,
    test_samples=None,
    output_dir=None
):
    """
    加载预训练模型和LoRA权重，并进行测试
    
    参数:
    - base_model_path: 基础模型路径
    - lora_weights_path: LoRA权重文件路径
    - lora_r: LoRA秩
    - lora_alpha: LoRA缩放因子
    - lora_dropout: LoRA dropout
    - target_modules: 应用LoRA的模块列表
    - qv_only: 是否只对Q和V应用LoRA
    - test_samples: 测试样本列表，如果为None则使用默认样本
    - output_dir: 输出目录，如果为None则使用lora_weights_path所在目录
    
    返回:
    - results: 测试结果列表
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(lora_weights_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志记录器
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_log_file = os.path.join(log_dir, f"test_{timestamp}.log")
    test_logger = setup_logger('test_logger', test_log_file)
    
    # 记录测试配置
    config = {
        'base_model_path': base_model_path,
        'lora_weights_path': lora_weights_path,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'target_modules': target_modules,
        'qv_only': qv_only
    }
    
    test_logger.info(f"测试配置: {json.dumps(config, ensure_ascii=False, indent=2)}")
    
    # 加载分词器
    test_logger.info("加载分词器...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 加载基础模型
    test_logger.info(f"加载基础模型: {base_model_path}")
    load_start_time = time.time()
    base_model = GPT.from_pretrained(base_model_path)
    load_time = time.time() - load_start_time
    test_logger.info(f"基础模型加载完成，耗时: {load_time:.2f}秒")
    
    # 创建LoRA配置
    lora_config = LoRAConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=target_modules,
        qv_only=qv_only
    )
    
    # 创建LoRA模型
    test_logger.info("创建LoRA模型...")
    model = LoRAGPT(base_model, lora_config)
    
    # 加载LoRA权重
    test_logger.info(f"加载LoRA权重: {lora_weights_path}")
    lora_load_start = time.time()
    success = model.load_lora_weights(lora_weights_path)
    lora_load_time = time.time() - lora_load_start
    
    if not success:
        test_logger.error("警告: 加载LoRA权重失败!")
        return []
    
    test_logger.info(f"LoRA权重加载完成，耗时: {lora_load_time:.2f}秒")
    
    # 测试模型
    results = test_model(model, tokenizer, test_samples, logger=test_logger)
    
    # 保存测试结果
    results_path = os.path.join(output_dir, f"test_results_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    test_logger.info(f"测试结果已保存至: {results_path}")
    
    return results, test_logger


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练或测试LoRA模型")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="运行模式：训练或测试")
    parser.add_argument("--base_model", type=str, default="./local_gpt2", help="基础模型路径")
    parser.add_argument("--data_path", type=str, default="./data/processed_squad_1000.json", help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/default_lora", help="输出目录")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=2, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--target_modules", type=str, default="c_attn,c_proj,c_fc", help="应用LoRA的模块列表，以逗号分隔")
    parser.add_argument("--qv_only", action="store_true", help="是否只对Q和V应用LoRA")
    parser.add_argument("--lora_weights", type=str, default=None, help="LoRA权重文件路径（测试模式下使用）")
    parser.add_argument("--log_performance_every", type=int, default=10, help="每隔多少步记录一次性能指标")
    
    args = parser.parse_args()
    
    # 解析目标模块列表
    target_modules = args.target_modules.split(",")
    
    if args.mode == "train":
        # 训练模式
        model, train_losses, val_losses, logger = train_default_lora(
            base_model_path=args.base_model,
            data_path=args.data_path,
            output_dir=args.output_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            target_modules=target_modules,
            qv_only=args.qv_only,
            log_performance_every=args.log_performance_every
        )
        
        # 加载分词器用于测试
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # 测试模型
        test_results = test_model(model, tokenizer, logger=logger)
        
        # 保存测试结果
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        logger.info("\n模型训练和测试完成!")
        logger.info(f"测试结果已保存至: {test_results_path}")
        logger.info("可以通过以下命令加载模型并进行测试:")
        logger.info(f"python train_default_lora.py --mode test --lora_weights {os.path.join(args.output_dir, 'gpt2_lora_final.npz')}")
    
    else:
        # 测试模式
        lora_weights_path = args.lora_weights or os.path.join(args.output_dir, "gpt2_lora_final.npz")
        
        # 加载模型并测试
        results, logger = load_and_test_model(
            base_model_path=args.base_model,
            lora_weights_path=lora_weights_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            qv_only=args.qv_only,
            output_dir=args.output_dir
        )
        
        logger.info("\n模型测试完成!") 
