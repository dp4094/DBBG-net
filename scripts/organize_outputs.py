"""
整理outputs目录

清理临时训练目录和日志文件，保留有用的模型检查点。
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def organize_outputs(outputs_dir: str = 'outputs', dry_run: bool = False):
    """
    整理outputs目录
    
    Args:
        outputs_dir: outputs目录路径
        dry_run: 是否为演练模式
    """
    outputs_path = Path(outputs_dir)
    
    if not outputs_path.exists():
        print(f"⚠️  目录不存在: {outputs_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"{'[演练模式] ' if dry_run else ''}开始整理outputs目录")
    print(f"{'='*60}\n")
    
    # 1. 删除临时训练目录（时间戳命名的目录）
    temp_dirs = []
    for item in outputs_path.iterdir():
        if item.is_dir() and item.name.startswith('2025-'):
            temp_dirs.append(item)
    
    if temp_dirs:
        print(f"📁 删除临时训练目录 ({len(temp_dirs)} 个):")
        for temp_dir in temp_dirs:
            if dry_run:
                print(f"[演练] 将删除目录: {temp_dir.name}")
            else:
                try:
                    shutil.rmtree(temp_dir)
                    print(f"✓ 已删除目录: {temp_dir.name}")
                except Exception as e:
                    print(f"✗ 删除失败: {temp_dir.name} - {e}")
    
    # 2. 删除临时训练日志
    log_files = list(outputs_path.glob('training_log_*.txt'))
    if log_files:
        print(f"\n📄 删除临时训练日志 ({len(log_files)} 个):")
        for log_file in log_files:
            if dry_run:
                print(f"[演练] 将删除文件: {log_file.name}")
            else:
                try:
                    log_file.unlink()
                    print(f"✓ 已删除文件: {log_file.name}")
                except Exception as e:
                    print(f"✗ 删除失败: {log_file.name} - {e}")
    
    # 3. 删除"新建文件夹"
    new_folder = outputs_path / '新建文件夹'
    if new_folder.exists():
        print(f"\n📁 删除临时文件夹:")
        if dry_run:
            print(f"[演练] 将删除目录: 新建文件夹")
        else:
            try:
                shutil.rmtree(new_folder)
                print(f"✓ 已删除目录: 新建文件夹")
            except Exception as e:
                print(f"✗ 删除失败: 新建文件夹 - {e}")
    
    # 4. 显示保留的目录
    kept_dirs = []
    for item in outputs_path.iterdir():
        if item.is_dir() and item.name in ['cluener', 'weibo', 'msra', 'peoplesdaily']:
            kept_dirs.append(item.name)
    
    if kept_dirs:
        print(f"\n✅ 保留的数据集目录:")
        for dir_name in sorted(kept_dirs):
            print(f"  - {dir_name}/")
    
    print(f"\n{'='*60}")
    print(f"整理完成！")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='整理outputs目录')
    parser.add_argument('--dry-run', action='store_true', help='演练模式，不实际删除')
    parser.add_argument('--outputs-dir', default='outputs', help='outputs目录路径')
    
    args = parser.parse_args()
    
    if not args.dry_run:
        response = input("\n确认执行整理？(y/N): ")
        if response.lower() != 'y':
            print("已取消整理操作")
            return
    
    organize_outputs(args.outputs_dir, args.dry_run)


if __name__ == '__main__':
    main()
