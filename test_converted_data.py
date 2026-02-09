#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试转换后的数据格式是否正确
"""

import json
import os
import sys
from collections import Counter

# 添加项目路径
sys.path.append('.')

from models.GlobalPointer import DataMaker
from transformers import AutoTokenizer


def test_data_format(dataset_path, dataset_name):
    """
    测试数据格式是否正确
    """
    print(f"\n=== 测试 {dataset_name} 数据集 ===")
    
    # 检查必要文件是否存在
    required_files = ['train.json', 'test.json', 'ent2id.json']
    for file in required_files:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            print(f"❌ 缺少文件: {file}")
            return False
        else:
            print(f"✓ 文件存在: {file}")
    
    # 加载实体映射
    ent2id_path = os.path.join(dataset_path, 'ent2id.json')
    with open(ent2id_path, 'r', encoding='utf-8') as f:
        ent2id = json.load(f)
    
    print(f"实体类型: {list(ent2id.keys())}")
    print(f"实体数量: {len(ent2id)}")
    
    # 测试训练数据
    train_path = os.path.join(dataset_path, 'train.json')
    test_path = os.path.join(dataset_path, 'test.json')
    
    for split_name, file_path in [('train', train_path), ('test', test_path)]:
        print(f"\n--- 测试 {split_name} 数据 ---")
        
        # 统计信息
        total_samples = 0
        total_entities = 0
        entity_counter = Counter()
        text_lengths = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 检查必要字段
                    if 'text' not in data or 'label' not in data:
                        print(f"❌ 第{line_num}行缺少必要字段")
                        continue
                    
                    text = data['text']
                    label = data['label']
                    
                    total_samples += 1
                    text_lengths.append(len(text))
                    
                    # 统计实体
                    for ent_type, entities in label.items():
                        if ent_type in ent2id:
                            for ent_text, spans in entities.items():
                                total_entities += len(spans)
                                entity_counter[ent_type] += len(spans)
                        else:
                            print(f"⚠️ 未知实体类型: {ent_type}")
                    
                    # 只显示前几个样本的详细信息
                    if line_num <= 3:
                        print(f"样本 {line_num}: {text[:50]}...")
                        print(f"  实体: {label}")
                
                except json.JSONDecodeError as e:
                    print(f"❌ 第{line_num}行JSON解析错误: {e}")
                except Exception as e:
                    print(f"❌ 第{line_num}行处理错误: {e}")
        
        print(f"\n{split_name} 数据统计:")
        print(f"  总样本数: {total_samples}")
        print(f"  总实体数: {total_entities}")
        print(f"  平均文本长度: {sum(text_lengths)/len(text_lengths):.1f}")
        print(f"  最大文本长度: {max(text_lengths)}")
        print(f"  实体类型分布: {dict(entity_counter)}")
    
    return True


def test_datamaker_compatibility(dataset_path, dataset_name):
    """
    测试与DataMaker的兼容性
    """
    print(f"\n=== 测试 {dataset_name} 与DataMaker的兼容性 ===")
    
    try:
        # 初始化tokenizer
        model_path = "./pretrained_models/chinese-roberta-wwm-ext"
        if not os.path.exists(model_path):
            model_path = "hfl/chinese-roberta-wwm-ext"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 创建DataMaker实例
        data_maker = DataMaker(tokenizer)
        
        # 加载少量测试数据
        train_path = os.path.join(dataset_path, 'train.json')
        ent2id_path = os.path.join(dataset_path, 'ent2id.json')
        
        with open(ent2id_path, 'r', encoding='utf-8') as f:
            ent2id = json.load(f)
        
        test_samples = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # 只测试前5个样本
                    break
                test_samples.append(json.loads(line.strip()))
        
        print(f"测试 {len(test_samples)} 个样本...")
        
        for i, sample in enumerate(test_samples):
            try:
                # 转换数据格式以适配DataMaker
                # DataMaker期望的格式: {'text': str, 'entity_list': [(start, end, ent_type)]}
                entity_list = []
                for ent_type, entities in sample['label'].items():
                    for ent_text, spans in entities.items():
                        for span in spans:
                            entity_list.append((span[0], span[1], ent_type))
                
                adapted_sample = {
                    'text': sample['text'],
                    'entity_list': entity_list
                }
                
                # 测试数据生成
                inputs = data_maker.generate_inputs(
                    [adapted_sample], 128, ent2id, "train"
                )
                
                # 获取第一个样本的结果
                inputs = inputs[0]
                
                print(f"样本 {i+1}:")
                print(f"  原文本: {sample['text'][:50]}...")
                
                # inputs是一个元组: (sample, input_ids, attention_mask, token_type_ids, labels)
                sample_data, input_ids, attention_mask, token_type_ids, labels = inputs
                
                print(f"  input_ids shape: {input_ids.shape}")
                print(f"  attention_mask shape: {attention_mask.shape}")
                print(f"  token_type_ids shape: {token_type_ids.shape}")
                if labels is not None:
                    print(f"  labels shape: {labels.shape}")
                    # 检查labels的有效性
                    positive_labels = (labels == 1).sum()
                    print(f"  正样本数量: {positive_labels}")
                else:
                    print(f"  labels: None (预测模式)")
                
            except Exception as e:
                print(f"❌ 样本 {i+1} 处理失败: {e}")
                return False
        
        print("✓ DataMaker兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ DataMaker兼容性测试失败: {e}")
        return False


def main():
    """
    主测试函数
    """
    print("🧪 开始测试转换后的数据集...")
    
    base_dir = "./datasets/converted_ner"
    datasets = {
        'MSRA': 'msra',
        'PeoplesDaily': 'peoplesdaily', 
        'Weibo': 'weibo'
    }
    
    all_passed = True
    
    for dataset_name, dataset_dir in datasets.items():
        dataset_path = os.path.join(base_dir, dataset_dir)
        
        if not os.path.exists(dataset_path):
            print(f"⚠️ 数据集目录不存在: {dataset_path}")
            continue
        
        # 测试数据格式
        format_ok = test_data_format(dataset_path, dataset_name)
        
        # 测试DataMaker兼容性
        compat_ok = test_datamaker_compatibility(dataset_path, dataset_name)
        
        if not (format_ok and compat_ok):
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 所有测试通过！数据转换成功！")
        print("\n📝 使用说明:")
        print("1. 修改 config.py 中的 data_home 路径")
        print("2. 例如使用MSRA数据集:")
        print("   data_home = './datasets/converted_ner/msra'")
        print("3. 运行训练脚本即可开始训练")
    else:
        print("❌ 部分测试失败，请检查数据转换")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)