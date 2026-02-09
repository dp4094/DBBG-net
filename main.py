"""
GlobalPointer NER 项目主入口

提供统一的命令行接口用于训练和评估。
"""

import sys
import argparse
from config import common


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='GlobalPointer中文命名实体识别',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练模型
  python main.py train --dataset cluener --epochs 20
  
  # 评估模型
  python main.py eval --model-dir ./outputs/best_model
  
  # 使用配置文件
  python main.py train  # 使用config.py中的配置
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'eval'],
        help='运行模式：train（训练）或 eval（评估）'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=common['exp_name'],
        choices=['cluener', 'weibo', 'msra', 'peoplesdaily'],
        help=f'数据集名称（默认: {common["exp_name"]}）'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        help='模型目录路径（评估模式必需）'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='训练轮数（覆盖config.py中的设置）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='批次大小（覆盖config.py中的设置）'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='学习率（覆盖config.py中的设置）'
    )
    
    args = parser.parse_args()
    
    # 更新配置
    common['exp_name'] = args.dataset
    common['run_type'] = args.mode
    
    if args.mode == 'train':
        # 导入并运行训练脚本
        print(f"\n{'='*60}")
        print(f"开始训练 - 数据集: {args.dataset}")
        print(f"{'='*60}\n")
        
        # 更新训练配置
        if args.epochs:
            from config import train_config
            train_config['hyper_parameters']['epochs'] = args.epochs
            print(f"训练轮数: {args.epochs}")
        
        if args.batch_size:
            from config import train_config
            train_config['hyper_parameters']['batch_size'] = args.batch_size
            print(f"批次大小: {args.batch_size}")
        
        if args.lr:
            from config import train_config
            train_config['hyper_parameters']['lr'] = args.lr
            print(f"学习率: {args.lr}")
        
        print()
        
        # 运行训练
        import train
        
    elif args.mode == 'eval':
        # 导入并运行评估脚本
        print(f"\n{'='*60}")
        print(f"开始评估 - 数据集: {args.dataset}")
        print(f"{'='*60}\n")
        
        # 更新评估配置
        if args.model_dir:
            from config import eval_config
            eval_config['model_state_dir'] = args.model_dir
            print(f"模型目录: {args.model_dir}\n")
        
        # 运行评估
        import evaluate


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
