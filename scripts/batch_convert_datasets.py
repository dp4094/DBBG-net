#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换datasets/NER下的所有BIOS格式数据集
"""

import os
import sys
from convert_bios_to_json import BIOSConverter


def batch_convert_datasets():
    """
    批量转换所有NER数据集
    """
    base_dir = "./datasets/NER"
    output_base_dir = "./datasets/converted_ner"
    
    # 确保输出目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    converter = BIOSConverter()
    
    # 定义数据集配置
    datasets_config = [
        {
            'name': 'MSRA',
            'type': 'msra',
            'files': {
                'train': 'msra_train_bio.txt',
                'test': 'msra_test_bio.txt'
            }
        },
        {
            'name': 'PeoplesDaily',
            'type': 'peoples_daily', 
            'files': {
                'train': 'example.train',
                'dev': 'example.dev',
                'test': 'example.test'
            }
        },
        {
            'name': 'Weibo',
            'type': 'weibo',
            'files': {
                'train': 'weiboNER_2nd_conll.train',
                'dev': 'weiboNER_2nd_conll.dev', 
                'test': 'weiboNER_2nd_conll.test'
            }
        }
    ]
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        dataset_type = dataset_config['type']
        dataset_dir = os.path.join(base_dir, dataset_name if dataset_name != 'PeoplesDaily' else "People's Daily")
        
        print(f"\n=== 转换 {dataset_name} 数据集 ===")
        
        if not os.path.exists(dataset_dir):
            print(f"警告: 数据集目录不存在: {dataset_dir}")
            continue
        
        # 创建输出目录
        output_dir = os.path.join(output_base_dir, dataset_name.lower())
        os.makedirs(output_dir, exist_ok=True)
        
        # 用于收集所有数据以生成统一的ent2id
        all_json_data = []
        
        # 转换每个文件
        for split_name, filename in dataset_config['files'].items():
            input_file = os.path.join(dataset_dir, filename)
            output_file = os.path.join(output_dir, f"{split_name}.json")
            
            if not os.path.exists(input_file):
                print(f"警告: 文件不存在: {input_file}")
                continue
            
            print(f"转换 {split_name} 数据...")
            
            try:
                # 转换数据（不生成ent2id文件，稍后统一生成）
                json_data, _ = converter.convert_dataset(input_file, output_file, None, dataset_type)
                all_json_data.extend(json_data)
                
                print(f"✓ {split_name} 转换完成: {len(json_data)} 个样本")
                
            except Exception as e:
                print(f"✗ {split_name} 转换失败: {str(e)}")
                continue
        
        # 生成统一的ent2id文件
        if all_json_data:
            ent2id_file = os.path.join(output_dir, "ent2id.json")
            ent2id = converter.create_entity_mapping(all_json_data)
            
            import json
            with open(ent2id_file, 'w', encoding='utf-8') as f:
                json.dump(ent2id, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 实体映射文件已生成: {ent2id_file}")
            print(f"  实体类型: {list(ent2id.keys())}")
            print(f"  总样本数: {len(all_json_data)}")
        
        print(f"=== {dataset_name} 数据集转换完成 ===")
    
    print("\n🎉 所有数据集转换完成！")
    print(f"转换后的数据保存在: {output_base_dir}")
    
    # 显示转换结果摘要
    print("\n📊 转换结果摘要:")
    for root, dirs, files in os.walk(output_base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    dataset_name = os.path.basename(root)
                    print(f"  {dataset_name}/{file}: {len(lines)} 个样本")
                except:
                    pass


def create_unified_config():
    """
    为转换后的数据集创建配置文件
    """
    output_base_dir = "./datasets/converted_ner"
    config_file = os.path.join(output_base_dir, "dataset_configs.py")
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("# 转换后的NER数据集配置文件\n")
        f.write("import os\n\n")
        
        for dataset_name in ['MSRA', 'PeoplesDaily', 'Weibo']:
            dataset_dir = os.path.join(output_base_dir, dataset_name.lower())
            if os.path.exists(dataset_dir):
                dataset_name_lower = dataset_name.lower()
                config_content = f'''
# {dataset_name} 数据集配置
{dataset_name_lower}_config = {{
    "data_home": "./datasets/converted_ner/{dataset_name_lower}",
    "train_data": "train.json",
    "valid_data": "dev.json" if os.path.exists("./datasets/converted_ner/{dataset_name_lower}/dev.json") else "test.json",
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "hyper_parameters": {{
        "lr": 5e-6,
        "batch_size": 12,
        "epochs": 20,
        "max_seq_len": 128,  # 根据数据集调整
        # 其他参数继承自默认配置
    }}
}}
'''
                f.write(config_content)
                f.write("\n")
    
    print(f"✓ 配置文件已生成: {config_file}")


if __name__ == '__main__':
    print("🚀 开始批量转换NER数据集...")
    batch_convert_datasets()
    create_unified_config()
    print("\n✅ 批量转换完成！")
    print("\n📝 使用说明:")
    print("1. 转换后的数据保存在 ./datasets/converted_ner/ 目录下")
    print("2. 每个数据集都有对应的 train.json, test.json 和 ent2id.json 文件")
    print("3. 可以修改 config.py 中的 data_home 路径来使用转换后的数据集")
    print("4. 示例: 将 data_home 改为 './datasets/converted_ner/msra' 来使用MSRA数据集")