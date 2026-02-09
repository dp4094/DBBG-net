#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用转换后的NER数据集的配置文件示例
"""

import os
from config import *  # 导入原始配置

# ===== MSRA数据集配置 =====
msra_config = {
    "data_home": "./datasets/converted_ner/msra",
    "train_data": "train.json",
    "valid_data": "test.json",  # MSRA没有dev集，使用test作为验证
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "hyper_parameters": {
        "lr": 5e-6,
        "batch_size": 12,
        "epochs": 20,
        "max_seq_len": 128,
        "seed": 42,
        # 其他参数继承自默认配置
    }
}

# ===== People's Daily数据集配置 =====
peoplesdaily_config = {
    "data_home": "./datasets/converted_ner/peoplesdaily",
    "train_data": "train.json",
    "valid_data": "dev.json",
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "hyper_parameters": {
        "lr": 5e-6,
        "batch_size": 12,
        "epochs": 20,
        "max_seq_len": 128,
        "seed": 42,
    }
}

# ===== Weibo数据集配置 =====
weibo_config = {
    "data_home": "./datasets/converted_ner/weibo",
    "train_data": "train.json",
    "valid_data": "dev.json",
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "hyper_parameters": {
        "lr": 5e-6,
        "batch_size": 16,  # Weibo数据集较小，可以用更大的batch size
        "epochs": 30,      # 数据量小，需要更多epochs
        "max_seq_len": 64, # Weibo文本较短，可以用更小的序列长度
        "seed": 42,
    }
}

# ===== 使用示例 =====
def get_config_for_dataset(dataset_name):
    """
    根据数据集名称获取对应的配置
    
    Args:
        dataset_name (str): 数据集名称 ('msra', 'peoplesdaily', 'weibo')
    
    Returns:
        dict: 数据集配置
    """
    configs = {
        'msra': msra_config,
        'peoplesdaily': peoplesdaily_config,
        'weibo': weibo_config
    }
    
    if dataset_name not in configs:
        raise ValueError(f"不支持的数据集: {dataset_name}. 支持的数据集: {list(configs.keys())}")
    
    return configs[dataset_name]


def update_train_config_for_dataset(dataset_name):
    """
    更新train_config以使用指定的数据集
    
    Args:
        dataset_name (str): 数据集名称
    
    Returns:
        dict: 更新后的train_config
    """
    dataset_config = get_config_for_dataset(dataset_name)
    
    # 复制原始train_config
    updated_config = train_config.copy()
    
    # 更新数据路径
    updated_config['data_home'] = dataset_config['data_home']
    updated_config['train_data'] = dataset_config['train_data']
    updated_config['valid_data'] = dataset_config['valid_data']
    updated_config['test_data'] = dataset_config['test_data']
    updated_config['ent2id'] = dataset_config['ent2id']
    
    # 更新超参数
    if 'hyper_parameters' in dataset_config:
        for key, value in dataset_config['hyper_parameters'].items():
            if key in updated_config:
                updated_config[key] = value
    
    return updated_config


# ===== 快速切换数据集的函数 =====
def switch_to_msra():
    """切换到MSRA数据集"""
    return update_train_config_for_dataset('msra')

def switch_to_peoplesdaily():
    """切换到People's Daily数据集"""
    return update_train_config_for_dataset('peoplesdaily')

def switch_to_weibo():
    """切换到Weibo数据集"""
    return update_train_config_for_dataset('weibo')


if __name__ == '__main__':
    # 使用示例
    print("=== NER数据集配置示例 ===")
    
    # 显示所有可用数据集
    datasets = ['msra', 'peoplesdaily', 'weibo']
    for dataset in datasets:
        config = get_config_for_dataset(dataset)
        print(f"\n{dataset.upper()} 数据集:")
        print(f"  数据路径: {config['data_home']}")
        print(f"  训练文件: {config['train_data']}")
        print(f"  验证文件: {config['valid_data']}")
        print(f"  测试文件: {config['test_data']}")
        print(f"  实体映射: {config['ent2id']}")
        
        # 检查文件是否存在
        data_home = config['data_home']
        if os.path.exists(data_home):
            print(f"  ✓ 数据目录存在")
            
            # 检查实体映射文件
            ent2id_path = os.path.join(data_home, config['ent2id'])
            if os.path.exists(ent2id_path):
                import json
                with open(ent2id_path, 'r', encoding='utf-8') as f:
                    ent2id = json.load(f)
                print(f"  实体类型: {list(ent2id.keys())}")
            else:
                print(f"  ❌ 实体映射文件不存在")
        else:
            print(f"  ❌ 数据目录不存在")
    
    print("\n=== 使用方法 ===")
    print("1. 在训练脚本中导入此配置文件:")
    print("   from config_ner_datasets import switch_to_msra")
    print("   train_config = switch_to_msra()")
    print("\n2. 或者直接修改config.py中的data_home路径:")
    print("   data_home = './datasets/converted_ner/msra'")
    print("\n3. 然后正常运行训练脚本即可")