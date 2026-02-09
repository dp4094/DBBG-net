#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试环境和代码是否能正常运行
"""

import sys
import os

def test_imports():
    """测试所有必要的导入"""
    print("=" * 60)
    print("测试Python包导入...")
    print("=" * 60)
    
    errors = []
    
    # 测试核心依赖
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name} 导入成功")
        except ImportError as e:
            print(f"✗ {name} 导入失败: {e}")
            errors.append(name)
    
    # 测试项目模块
    print("\n测试项目模块...")
    try:
        import config
        print("✓ config.py 导入成功")
    except Exception as e:
        print(f"✗ config.py 导入失败: {e}")
        errors.append('config')
    
    try:
        from models.GlobalPointer import GlobalPointer, DataMaker, MetricsCalculator
        print("✓ models/GlobalPointer.py 导入成功")
    except Exception as e:
        print(f"✗ models/GlobalPointer.py 导入失败: {e}")
        errors.append('GlobalPointer')
    
    try:
        from common.utils import Preprocessor
        print("✓ common/utils.py 导入成功")
    except Exception as e:
        print(f"✗ common/utils.py 导入失败: {e}")
        errors.append('utils')
    
    return len(errors) == 0, errors


def test_config():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试配置文件...")
    print("=" * 60)
    
    try:
        import config
        
        print(f"实验名称: {config.common['exp_name']}")
        print(f"编码器: {config.common['encoder']}")
        print(f"数据目录: {config.common['data_home']}")
        print(f"模型路径: {config.common['bert_path']}")
        print(f"运行类型: {config.common['run_type']}")
        
        # 检查数据集路径
        data_home = config.common['data_home']
        exp_name = config.common['exp_name']
        
        if exp_name:
            dataset_path = os.path.join(data_home, exp_name)
        else:
            dataset_path = data_home
        
        print(f"\n数据集路径: {dataset_path}")
        
        if os.path.exists(dataset_path):
            print(f"✓ 数据集目录存在")
            
            # 检查必要文件
            required_files = ['train.json', 'dev.json', 'test.json', 'ent2id.json']
            for file in required_files:
                file_path = os.path.join(dataset_path, file)
                if os.path.exists(file_path):
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ {file} (不存在)")
        else:
            print(f"✗ 数据集目录不存在: {dataset_path}")
        
        # 检查预训练模型
        bert_path = config.common['bert_path']
        print(f"\n预训练模型路径: {bert_path}")
        if os.path.exists(bert_path):
            print(f"✓ 预训练模型目录存在")
        else:
            print(f"✗ 预训练模型目录不存在")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("测试数据加载...")
    print("=" * 60)
    
    try:
        import json
        import config
        
        data_home = config.common['data_home']
        exp_name = config.common['exp_name']
        
        if exp_name:
            dataset_path = os.path.join(data_home, exp_name)
        else:
            dataset_path = data_home
        
        # 加载ent2id
        ent2id_path = os.path.join(dataset_path, 'ent2id.json')
        if os.path.exists(ent2id_path):
            with open(ent2id_path, 'r', encoding='utf-8') as f:
                ent2id = json.load(f)
            print(f"✓ 实体映射加载成功")
            print(f"  实体类型: {list(ent2id.keys())}")
            print(f"  实体数量: {len(ent2id)}")
        else:
            print(f"✗ 实体映射文件不存在: {ent2id_path}")
            return False
        
        # 加载训练数据（只读取前几行）
        train_path = os.path.join(dataset_path, 'train.json')
        if os.path.exists(train_path):
            with open(train_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    lines.append(json.loads(line))
            
            print(f"✓ 训练数据加载成功")
            print(f"  样本示例: {lines[0]['text'][:50]}...")
            print(f"  实体示例: {list(lines[0]['label'].keys())}")
        else:
            print(f"✗ 训练数据文件不存在: {train_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试模型创建...")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoTokenizer, BertModel
        from models.GlobalPointer import GlobalPointer
        
        # 使用简单的BERT模型测试
        print("创建简单的测试模型...")
        
        # 创建一个小的BERT配置用于测试
        from transformers import BertConfig
        config = BertConfig(
            vocab_size=21128,
            hidden_size=768,
            num_hidden_layers=2,  # 只用2层测试
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512
        )
        
        encoder = BertModel(config)
        model = GlobalPointer(
            encoder=encoder,
            ent_type_size=10,  # CLUENER有10种实体类型
            inner_dim=64,
            gru_hidden_size=320,
            dropout=0.1
        )
        
        print(f"✓ 模型创建成功")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        print("\n测试前向传播...")
        batch_size = 2
        seq_len = 128
        
        input_ids = torch.randint(0, 21128, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_ids, attention_mask, token_type_ids)
        
        print(f"✓ 前向传播成功")
        print(f"  输出形状: {output.shape}")
        print(f"  预期形状: (batch_size={batch_size}, ent_type_size=10, seq_len={seq_len}, seq_len={seq_len})")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "🧪" * 30)
    print("GlobalPointer环境测试")
    print("🧪" * 30 + "\n")
    
    results = []
    
    # 测试导入
    success, errors = test_imports()
    results.append(("包导入", success))
    
    if not success:
        print(f"\n⚠️  缺少以下依赖包: {', '.join(errors)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 测试配置
    success = test_config()
    results.append(("配置文件", success))
    
    # 测试数据加载
    success = test_data_loading()
    results.append(("数据加载", success))
    
    # 测试模型创建
    success = test_model_creation()
    results.append(("模型创建", success))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！环境配置正确，可以开始训练。")
        print("\n下一步:")
        print("1. 确保数据集已准备好")
        print("2. 确保预训练模型已下载")
        print("3. 修改config.py中的配置")
        print("4. 运行: python train.py")
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
