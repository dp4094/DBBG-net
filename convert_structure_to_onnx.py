#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GlobalPointer模型结构转换为ONNX格式
只转换模型结构，不加载预训练权重
适用于DeBERTa训练的模型但环境不支持DeBERTa的情况
"""

import torch
import torch.onnx
import os
import sys
import json
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 添加模型路径
sys.path.append('./models')
from GlobalPointer import GlobalPointer

class ONNXConverter:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def load_entity_types(self):
        """加载实体类型"""
        ent2id_path = os.path.join('datasets', 'converted_ner', 'ent2id.json')
        if not os.path.exists(ent2id_path):
            ent2id_path = os.path.join('datasets', 'cluener', 'ent2id.json')
        
        if os.path.exists(ent2id_path):
            with open(ent2id_path, 'r', encoding='utf-8') as f:
                ent2id = json.load(f)
            return list(ent2id.keys()), len(ent2id)
        else:
            # 默认CLUENER实体类型
            default_entities = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
            return default_entities, len(default_entities)
    
    def create_model_structure(self, num_classes):
        """创建模型结构（不加载权重）"""
        print("创建模型结构...")
        
        # 使用DeBERTa-v2作为编码器结构
        try:
            deberta_model_path = os.path.join('pretrained_models', 'Erlangshen-DeBERTa-v2-320M-Chinese')
            if os.path.exists(deberta_model_path):
                print(f"使用本地DeBERTa-v2模型结构: {deberta_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(deberta_model_path)
                encoder = AutoModel.from_pretrained(deberta_model_path)
            else:
                # 备选方案：使用在线DeBERTa模型
                print("使用在线DeBERTa模型结构: microsoft/deberta-v3-base")
                self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
                encoder = AutoModel.from_pretrained('microsoft/deberta-v3-base')
                
            print(f"编码器配置: hidden_size={encoder.config.hidden_size}, num_layers={encoder.config.num_hidden_layers}")
        except Exception as e:
            print(f"加载DeBERTa模型结构失败: {e}")
            # 如果DeBERTa加载失败，回退到BERT
            print("回退到BERT模型...")
            try:
                local_bert_path = os.path.join('pretrained_models', 'bert-base-chinese')
                if os.path.exists(local_bert_path):
                    print(f"使用本地BERT模型结构: {local_bert_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(local_bert_path)
                    encoder = AutoModel.from_pretrained(local_bert_path)
                else:
                    print("使用在线BERT模型结构: bert-base-chinese")
                    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
                    encoder = AutoModel.from_pretrained('bert-base-chinese')
                print(f"编码器配置: hidden_size={encoder.config.hidden_size}, num_layers={encoder.config.num_hidden_layers}")
            except Exception as bert_e:
                print(f"BERT模型加载也失败: {bert_e}")
                raise bert_e
        
        # 创建GlobalPointer模型（使用随机初始化权重）
        model = GlobalPointer(
            encoder=encoder,
            ent_type_size=num_classes,
            inner_dim=64,
            RoPE=True
        )
        
        model.eval()
        model.to(self.device)
        
        print(f"模型结构创建完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def convert_to_onnx(self, model, output_path, max_seq_len=512):
        """转换模型为ONNX格式"""
        print(f"开始转换为ONNX格式，最大序列长度: {max_seq_len}")
        
        # 创建虚拟输入
        batch_size = 1
        dummy_input_ids = torch.randint(0, self.tokenizer.vocab_size, (batch_size, max_seq_len), dtype=torch.long).to(self.device)
        dummy_attention_mask = torch.ones((batch_size, max_seq_len), dtype=torch.long).to(self.device)
        dummy_token_type_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(self.device)
        
        # 输入名称
        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        output_names = ['logits']
        
        # 动态轴配置
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'token_type_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence', 2: 'sequence'}
        }
        
        try:
            # 导出ONNX模型
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            print(f"✅ ONNX模型已保存到: {output_path}")
            print(f"文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX转换失败: {e}")
            return False
    
    def verify_onnx_model(self, onnx_path, max_seq_len=512):
        """验证ONNX模型"""
        try:
            import onnxruntime as ort
            
            print("验证ONNX模型...")
            
            # 创建ONNX Runtime会话
            session = ort.InferenceSession(onnx_path)
            
            # 创建测试输入
            batch_size = 1
            test_input_ids = np.random.randint(0, self.tokenizer.vocab_size, (batch_size, max_seq_len), dtype=np.int64)
            test_attention_mask = np.ones((batch_size, max_seq_len), dtype=np.int64)
            test_token_type_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            
            # 运行推理
            inputs = {
                'input_ids': test_input_ids,
                'attention_mask': test_attention_mask,
                'token_type_ids': test_token_type_ids
            }
            
            outputs = session.run(None, inputs)
            
            print(f"✅ ONNX模型验证成功")
            print(f"输出形状: {outputs[0].shape}")
            
            return True
            
        except ImportError:
            print("⚠️  未安装onnxruntime，跳过验证")
            return True
        except Exception as e:
            print(f"❌ ONNX模型验证失败: {e}")
            return False

def main():
    print("=" * 60)
    print("GlobalPointer模型结构ONNX转换器")
    print("（不加载预训练权重，仅转换模型结构）")
    print("=" * 60)
    
    converter = ONNXConverter()
    
    # 加载实体类型
    entity_types, num_classes = converter.load_entity_types()
    print(f"实体类别数量: {num_classes}")
    print(f"实体类别: {entity_types}")
    
    # 创建模型结构
    model = converter.create_model_structure(num_classes)
    
    # 转换为ONNX
    output_path = "globalpointer_structure.onnx"
    success = converter.convert_to_onnx(model, output_path)
    
    if success:
        # 验证ONNX模型
        converter.verify_onnx_model(output_path)
        
        print("\n=" * 60)
        print("转换完成！")
        print(f"ONNX模型文件: {output_path}")
        print("注意: 此模型使用随机初始化权重，需要重新训练或加载对应权重")
        print("=" * 60)
    else:
        print("\n❌ 转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main()