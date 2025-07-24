"""
下载NLTK资源的辅助脚本
运行此脚本以下载评估所需的NLTK资源
"""

import nltk
import os
import sys

def download_nltk_resources():
    """下载评估所需的NLTK资源"""
    print("开始下载NLTK资源...")
    
    # 设置NLTK数据目录
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # 下载资源
    resources = ['punkt', 'wordnet', 'omw-1.4']
    for resource in resources:
        print(f"下载 {resource}...")
        try:
            nltk.download(resource, quiet=False)
            print(f"{resource} 下载成功")
        except Exception as e:
            print(f"下载 {resource} 时出错: {e}")
    
    # 验证资源是否已下载
    print("\n验证资源...")
    all_success = True
    
    # 资源路径映射
    resource_paths = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    
    for resource, path in resource_paths.items():
        try:
            if nltk.data.find(path):
                print(f"{resource}: 已找到")
            else:
                print(f"{resource}: 未找到")
                all_success = False
        except LookupError:
            print(f"{resource}: 未找到")
            all_success = False
    
    if all_success:
        print("\n所有NLTK资源已成功下载！")
    else:
        print("\n警告: 部分资源可能未成功下载。")
        print("如果评估脚本出错，请尝试手动下载:")
        print("python -c \"import nltk; nltk.download()\"")
    
    return all_success

if __name__ == "__main__":
    download_nltk_resources() 