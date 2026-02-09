"""
配置文件测试

验证config.py中的配置是否正确
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "="*60)
    print("测试配置文件加载")
    print("="*60)
    
    try:
        import config
        print("✓ 配置文件导入成功")
        
        # 检查基本配置
        assert hasattr(config, 'common'), "缺少 common 配置"
        assert hasattr(config, 'train_config'), "缺少 train_config 配置"
        assert hasattr(config, 'eval_config'), "缺少 eval_config 配置"
        print("✓ 配置结构正确")
        
        # 检查关键配置项
        assert 'exp_name' in config.common, "缺少 exp_name 配置"
        assert 'run_type' in config.common, "缺少 run_type 配置"
        assert 'data_home' in config.common, "缺少 data_home 配置"
        assert 'bert_path' in config.common, "缺少 bert_path 配置"
        print("✓ 关键配置项存在")
        
        # 打印当前配置
        print(f"\n当前配置:")
        print(f"  - 数据集: {config.common['exp_name']}")
        print(f"  - 运行模式: {config.common['run_type']}")
        print(f"  - 数据目录: {config.common['data_home']}")
        print(f"  - 模型路径: {config.common['bert_path']}")
        
        return True
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_paths():
    """测试数据集路径"""
    print("\n" + "="*60)
    print("测试数据集路径")
    print("="*60)
    
    try:
        import config
        
        # 测试 cluener 数据集
        datasets = ['cluener', 'weibo']
        
        for dataset_name in datasets:
            print(f"\n检查数据集: {dataset_name}")
            dataset_path = os.path.join(config.common['data_home'], dataset_name)
            
            if os.path.exists(dataset_path):
                print(f"  ✓ 数据集目录存在: {dataset_path}")
                
                # 检查必需文件
                required_files = ['train.json', 'dev.json', 'test.json', 'ent2id.json']
                for file_name in required_files:
                    file_path = os.path.join(dataset_path, file_name)
                    if os.path.exists(file_path):
                        print(f"    ✓ {file_name} 存在")
                    else:
                        print(f"    ✗ {file_name} 不存在")
            else:
                print(f"  ✗ 数据集目录不存在: {dataset_path}")
        
        return True
    except Exception as e:
        print(f"✗ 数据集路径测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_paths():
    """测试模型路径"""
    print("\n" + "="*60)
    print("测试预训练模型路径")
    print("="*60)
    
    try:
        import config
        
        bert_path = config.common['bert_path']
        print(f"预训练模型路径: {bert_path}")
        
        if os.path.exists(bert_path):
            print(f"✓ 模型目录存在")
            
            # 检查必需文件
            required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
            for file_name in required_files:
                file_path = os.path.join(bert_path, file_name)
                if os.path.exists(file_path):
                    print(f"  ✓ {file_name} 存在")
                else:
                    print(f"  ✗ {file_name} 不存在")
        else:
            print(f"✗ 模型目录不存在")
            print(f"  请下载预训练模型到: {bert_path}")
            print(f"  参考: docs/installation.md")
        
        return True
    except Exception as e:
        print(f"✗ 模型路径测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_dirs():
    """测试输出目录"""
    print("\n" + "="*60)
    print("测试输出目录")
    print("="*60)
    
    try:
        import config
        
        # 检查输出目录
        output_dir = config.train_config.get('path_to_save_model', './outputs')
        print(f"训练输出目录: {output_dir}")
        
        if not os.path.exists(output_dir):
            print(f"  ⚠️  输出目录不存在，将在训练时自动创建")
        else:
            print(f"  ✓ 输出目录存在")
        
        # 检查结果目录
        results_dir = config.eval_config.get('save_res_dir', './results')
        print(f"评估结果目录: {results_dir}")
        
        if not os.path.exists(results_dir):
            print(f"  ⚠️  结果目录不存在，将在评估时自动创建")
        else:
            print(f"  ✓ 结果目录存在")
        
        return True
    except Exception as e:
        print(f"✗ 输出目录测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("GlobalPointer 配置测试")
    print("="*60)
    
    tests = [
        ("配置文件加载", test_config_loading),
        ("数据集路径", test_dataset_paths),
        ("预训练模型路径", test_model_paths),
        ("输出目录", test_output_dirs),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{test_name}' 出现异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！配置正确。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查配置。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
