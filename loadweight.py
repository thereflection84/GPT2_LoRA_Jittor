import numpy as np
from transformers import GPT2LMHeadModel
import os
import argparse

def convert_gpt2_weights_to_npz(model_path, output_path=None):
    """
    将HuggingFace的GPT2模型权重转换为npz格式，以便Jittor模型加载
    
    参数:
        model_path: HuggingFace模型路径，例如'./local_gpt2'
        output_path: 输出的npz文件路径，默认为model_path同级目录下的gpt2_weights.npz
    """
    print(f"从{model_path}加载模型...")
    
    print("尝试直接从safetensors文件加载...")
        
    try:
        # 尝试使用safetensors直接加载
        from safetensors import safe_open
        import glob
            
        # 查找safetensors文件
        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError("未找到safetensors文件")

        # 加载第一个找到的safetensors文件
        safetensor_file = safetensor_files[0]
        print(f"从{safetensor_file}加载权重...")
            
        weights_dict = {}
        with safe_open(safetensor_file, framework="numpy") as f:
            for k in f.keys():
                weights_dict[k] = f.get_tensor(k)
    except Exception as inner_e:
        print(f"无法从safetensors加载: {inner_e}")

    
    # 如果没有指定输出路径，则在模型路径同级目录下创建
    if output_path is None:
        output_path = os.path.join(os.path.dirname(model_path), "gpt2_weights.npz")
    
    # 保存为npz文件
    np.savez(output_path, **weights_dict)
    print(f"权重已保存至: {output_path}")
    
    # 打印权重信息
    print("\n权重信息:")
    print(f"总参数数量: {len(weights_dict)}")
    total_params = sum(arr.size for arr in weights_dict.values())
    print(f"总参数大小: {total_params:,} 个参数")
    
    # 打印部分关键权重的形状
    key_weights = [
        "transformer.wte.weight",  # 词嵌入
        "transformer.wpe.weight",  # 位置嵌入
        "transformer.h.0.attn.c_attn.weight",  # 第一层注意力权重
        "transformer.h.0.attn.c_proj.weight",  # 第一层注意力投影
        "transformer.h.0.mlp.c_fc.weight",  # 第一层MLP
        "transformer.ln_f.weight",  # 最终层归一化
        "lm_head.weight"  # 语言模型头
    ]
    
    print("\n关键权重形状:")
    for key in key_weights:
        if key in weights_dict:
            print(f"{key}: {weights_dict[key].shape}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将HuggingFace的GPT2模型权重转换为npz格式")
    parser.add_argument("--model_path", type=str, default="./local_gpt2", 
                        help="HuggingFace模型路径，例如'./local_gpt2'")
    parser.add_argument("--output_path", type=str, default=None, 
                        help="输出的npz文件路径，默认为model_path同级目录下的gpt2_weights.npz")
    
    args = parser.parse_args()
    convert_gpt2_weights_to_npz(args.model_path, args.output_path)
