import numpy as np
import torch
import re
import json
import os
from transformers import GPT2Tokenizer


# 生成回答
def generate_answer(model, tokenizer, context, question, max_length=50):
    """
    使用模型生成问题的回答
    
    参数:
    - model: GPT或LoRAGPT模型
    - tokenizer: GPT2分词器
    - context: 上下文文本
    - question: 问题文本
    - max_length: 生成答案的最大长度
    
    返回:
    - answer: 生成的回答文本
    """
    model.eval()
    
    # 准备输入
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    print(f"输入提示: {input_text}")
    input_ids = tokenizer.encode(input_text)
    print(f"输入ID长度: {len(input_ids)}")
    
    # 生成回答 - 使用简化的贪婪解码
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    
    # 移动到正确的设备
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 逐步生成
    for step in range(max_length):
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # 获取下一个token的logits
        next_token_logits = outputs[0, -1, :].detach().cpu().numpy()
        
        # 简单的贪婪解码 - 选择概率最高的token
        next_token_id = int(np.argmax(next_token_logits))
        
        # 添加到序列
        input_ids.append(next_token_id)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # 打印生成的token
        if step < 5 or step > max_length - 5:
            print(f"步骤 {step}: 生成token {next_token_id} ({tokenizer.decode([next_token_id])})")
        
        # 如果生成了结束符或达到最大长度，则停止
        if next_token_id == tokenizer.eos_token_id:
            break
    
    # 解码生成的文本
    generated_text = tokenizer.decode(input_ids)
    print(f"完整生成文本: {generated_text}")
    
    # 提取回答部分
    answer = generated_text.split("Answer:")[-1].strip()
    print(f"提取的回答: {answer}")
    
    return answer


# 数据加载器
class SQuADDataLoader:
    def __init__(self, data_path, tokenizer, block_size=1024, batch_size=4, shuffle=True):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        # 转换为PyTorch张量
        x_tensor = torch.tensor(np.array(x), dtype=torch.long).to(self.device)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long).to(self.device)
        
        return x_tensor, y_tensor


# 评估函数
def evaluate(model, dataloader, num_batches=10):
    model.eval()
    losses = []
    
    for _ in range(num_batches):
        x_batch, y_batch = dataloader.get_batch('val')
        with torch.no_grad():
            _, loss = model(x_batch, y_batch)
        losses.append(loss.item())
    
    return sum(losses) / len(losses)


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
