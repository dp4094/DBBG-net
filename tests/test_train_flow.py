"""
训练流程测试

快速测试训练流程是否正常（只训练1个batch）
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_train_flow():
    """测试训练流程"""
    print("\n" + "="*60)
    print("测试训练流程（快速测试）")
    print("="*60)
    
    try:
        # 导入必要的模块
        import torch
        import config as config_module
        from transformers import AutoTokenizer, DebertaV2Model
        from models.GlobalPointer import DataMaker, MyDataset, GlobalPointer
        from torch.utils.data import DataLoader
        import json
        
        print("✓ 模块导入成功")
        
        # 设置为训练模式
        config_module.common['run_type'] = 'train'
        config = config_module.train_config
        hyper_parameters = config["hyper_parameters"]
        
        # 设置设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"✓ 使用设备: {device}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["bert_path"], do_lower_case=False)
        print("✓ Tokenizer加载成功")
        
        # 加载实体映射
        ent2id_path = os.path.join(
            config_module.common["data_home"], 
            config_module.common["exp_name"], 
            config["ent2id"]
        )
        with open(ent2id_path, encoding="utf-8") as f:
            ent2id = json.load(f)
        ent_type_size = len(ent2id)
        print(f"✓ 实体映射加载成功，实体类型数: {ent_type_size}")
        
        # 加载少量训练数据
        train_data_path = os.path.join(
            config_module.common["data_home"], 
            config_module.common["exp_name"], 
            config["train_data"]
        )
        
        train_data = []
        with open(train_data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 4:  # 只加载4条数据用于测试
                    break
                line = json.loads(line)
                item = {}
                item["text"] = line["text"]
                item["entity_list"] = []
                for k, v in line['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entity_list"].append((start, end, k))
                train_data.append(item)
        
        print(f"✓ 加载测试数据: {len(train_data)} 条")
        
        # 创建DataLoader
        data_maker = DataMaker(tokenizer)
        train_dataloader = DataLoader(
            MyDataset(train_data),
            batch_size=2,  # 小批次用于测试
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=lambda x: data_maker.generate_batch(x, hyper_parameters["max_seq_len"], ent2id)
        )
        print("✓ DataLoader创建成功")
        
        # 创建模型
        encoder = DebertaV2Model.from_pretrained(config["bert_path"], ignore_mismatched_sizes=True)
        model = GlobalPointer(
            encoder, 
            ent_type_size, 
            64,
            gru_hidden_size=hyper_parameters.get("gru_hidden_size", 320),
            dropout=hyper_parameters.get("dropout", 0.1),
            use_layernorm=hyper_parameters.get("use_layernorm", True)
        )
        model = model.to(device)
        print("✓ 模型创建成功")
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(hyper_parameters["lr"]), 
            weight_decay=hyper_parameters.get("weight_decay", 0.01)
        )
        print("✓ 优化器创建成功")
        
        # 测试一个训练步骤
        model.train()
        batch_data = next(iter(train_dataloader))
        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_token_type_ids = batch_token_type_ids.to(device)
        batch_labels = batch_labels.to(device)
        
        print("✓ 数据移动到设备成功")
        
        # 前向传播
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        print(f"✓ 前向传播成功，输出形状: {logits.shape}")
        
        # 计算损失
        from common.utils import multilabel_categorical_crossentropy
        loss = multilabel_categorical_crossentropy(batch_labels, logits)
        print(f"✓ 损失计算成功，损失值: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        print("✓ 反向传播成功")
        
        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()
        print("✓ 优化器更新成功")
        
        print("\n" + "="*60)
        print("🎉 训练流程测试通过！")
        print("="*60)
        print("\n提示:")
        print("  - 配置正确，可以开始完整训练")
        print("  - 训练命令: python train.py")
        print("  - 或使用: python main.py train --dataset weibo")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 训练流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_train_flow()
    sys.exit(0 if success else 1)
