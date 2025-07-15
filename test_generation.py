import os
import jittor as jt
import numpy as np
from transformers import GPT2Tokenizer
import argparse

# 显式启用GPU
jt.flags.use_cuda = 1

# 导入自定义模块
from GPT2_jittor import GPT
from LoRA import LoRAConfig
from lora_models import LoRAGPT
from model_utils import generate_answer

def test_base_model(model_path='./local_gpt2'):
    """测试基础模型的生成能力"""
    print("\n" + "="*50)
    print(f"测试基础模型: {model_path}")
    print("="*50)
    
    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 加载基础模型
    base_model = GPT.from_pretrained(model_path)
    
    # 测试样本
    test_samples = [
        {
            "context": "Jittor is a high-performance deep learning framework based on Just-in-time compilation technology, developed by the Computer Science Department at Tsinghua University.",
            "question": "What is Jittor?"
        },
        {
            "context": "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that works by freezing the pre-trained model weights and injecting trainable low-rank matrices into each layer of the Transformer architecture.",
            "question": "What is the main purpose of LoRA?"
        }
    ]
    
    # 测试生成
    for i, sample in enumerate(test_samples):
        print(f"\n测试样本 {i+1}:")
        print(f"上下文: {sample['context']}")
        print(f"问题: {sample['question']}")
        
        # 使用简单的生成方式
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        input_ids = tokenizer.encode(prompt)
        input_tensor = jt.array(np.array(input_ids)).reshape(1, -1)
        
        # 逐步生成
        max_length = 30
        for step in range(max_length):
            with jt.no_grad():
                outputs = base_model(input_tensor)
            
            # 获取下一个token的logits
            next_token_logits = outputs[0, -1, :].numpy()
            
            # 贪婪解码
            next_token_id = int(np.argmax(next_token_logits))
            
            # 添加到序列
            input_ids.append(next_token_id)
            input_tensor = jt.array(np.array(input_ids)).reshape(1, -1)
            
            # 打印生成的token
            if step < 5:
                print(f"步骤 {step}: 生成token {next_token_id} ({tokenizer.decode([next_token_id])})")
            
            # 如果生成了结束符或达到最大长度，则停止
            if next_token_id == tokenizer.eos_token_id:
                break
        
        # 解码生成的文本
        generated_text = tokenizer.decode(input_ids)
        print(f"完整生成文本: {generated_text}")
        
        # 提取回答部分
        answer = generated_text.split("Answer:")[-1].strip()
        print(f"生成的回答: {answer}")
        print("-"*50)

def test_lora_model(
    base_model_path='./local_gpt2',
    lora_weights_path='./checkpoints/default_lora/gpt2_lora_final.npz',
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj", "c_fc"],
    qv_only=False
):
    """测试LoRA模型的生成能力"""
    print("\n" + "="*50)
    print(f"测试LoRA模型: {lora_weights_path}")
    print("="*50)
    
    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 加载基础模型
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
    model = LoRAGPT(base_model, lora_config)
    
    # 加载LoRA权重
    print(f"加载LoRA权重: {lora_weights_path}")
    success = model.load_lora_weights(lora_weights_path)
    
    if not success:
        print("警告: 加载LoRA权重失败!")
        return
    
    # 测试样本
    test_samples = [
        {
            "context": "Jittor is a high-performance deep learning framework based on Just-in-time compilation technology, developed by the Computer Science Department at Tsinghua University.",
            "question": "What is Jittor?"
        },
        {
            "context": "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that works by freezing the pre-trained model weights and injecting trainable low-rank matrices into each layer of the Transformer architecture.",
            "question": "What is the main purpose of LoRA?"
        }
    ]
    
    # 测试生成
    for i, sample in enumerate(test_samples):
        print(f"\n测试样本 {i+1}:")
        print(f"上下文: {sample['context']}")
        print(f"问题: {sample['question']}")
        
        # 使用模型工具中的generate_answer函数
        answer = generate_answer(model, tokenizer, sample["context"], sample["question"], max_length=30)
        print(f"生成的回答: {answer}")
        print("-"*50)
        
        # 尝试直接使用模型生成
        print("使用直接生成方式:")
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        input_ids = tokenizer.encode(prompt)
        input_tensor = jt.array(np.array(input_ids)).reshape(1, -1)
        
        # 逐步生成
        max_length = 30
        for step in range(max_length):
            with jt.no_grad():
                outputs = model(input_tensor)
            
            # 获取下一个token的logits
            next_token_logits = outputs[0, -1, :].numpy()
            
            # 打印前5个最高概率的token
            if step < 3:
                top5_indices = np.argsort(-next_token_logits)[:5]
                print(f"步骤 {step} 的前5个token:")
                for idx in top5_indices:
                    print(f"  Token {idx}: {tokenizer.decode([idx])} (概率: {np.exp(next_token_logits[idx] - np.max(next_token_logits)):.4f})")
            
            # 贪婪解码
            next_token_id = int(np.argmax(next_token_logits))
            
            # 添加到序列
            input_ids.append(next_token_id)
            input_tensor = jt.array(np.array(input_ids)).reshape(1, -1)
            
            # 打印生成的token
            if step < 5:
                print(f"步骤 {step}: 生成token {next_token_id} ({tokenizer.decode([next_token_id])})")
            
            # 如果生成了结束符或达到最大长度，则停止
            if next_token_id == tokenizer.eos_token_id:
                break
        
        # 解码生成的文本
        generated_text = tokenizer.decode(input_ids)
        print(f"完整生成文本: {generated_text}")
        
        # 提取回答部分
        answer = generated_text.split("Answer:")[-1].strip()
        print(f"生成的回答: {answer}")
        print("-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试模型生成能力")
    parser.add_argument("--mode", type=str, choices=["base", "lora", "both"], default="both", help="测试模式: 基础模型、LoRA模型或两者")
    parser.add_argument("--base_model", type=str, default="./local_gpt2", help="基础模型路径")
    parser.add_argument("--lora_weights", type=str, default="./checkpoints/default_lora/gpt2_lora_final.npz", help="LoRA权重文件路径")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA缩放因子")
    parser.add_argument("--target_modules", type=str, default="c_attn,c_proj,c_fc", help="应用LoRA的模块列表，以逗号分隔")
    parser.add_argument("--qv_only", action="store_true", help="是否只对Q和V应用LoRA")
    
    args = parser.parse_args()
    
    # 解析目标模块列表
    target_modules = args.target_modules.split(",")
    
    # 测试模型
    if args.mode in ["base", "both"]:
        test_base_model(args.base_model)
    
    if args.mode in ["lora", "both"]:
        test_lora_model(
            base_model_path=args.base_model,
            lora_weights_path=args.lora_weights,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            qv_only=args.qv_only
        ) 
