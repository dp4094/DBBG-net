"""
整理数据集目录

将未使用的数据集移动到archive目录，保持数据集目录结构清晰。
"""

import os
import shutil
from pathlib import Path


def organize_datasets(datasets_dir: str = 'datasets', dry_run: bool = False):
    """
    整理数据集目录
    
    Args:
        datasets_dir: 数据集目录路径
        dry_run: 是否为演练模式
    """
    datasets_path = Path(datasets_dir)
    
    if not datasets_path.exists():
        print(f"⚠️  目录不存在: {datasets_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"{'[演练模式] ' if dry_run else ''}开始整理数据集目录")
    print(f"{'='*60}\n")
    
    # 创建archive目录
    archive_dir = datasets_path / 'archive'
    if not dry_run and not archive_dir.exists():
        archive_dir.mkdir()
        print(f"✓ 创建归档目录: {archive_dir.relative_to(datasets_path.parent)}")
    elif dry_run:
        print(f"[演练] 将创建归档目录: archive/")
    
    # 移动未使用的原始数据集目录
    ner_dir = datasets_path / 'NER'
    if ner_dir.exists():
        print(f"\n📁 归档原始NER数据集目录:")
        if dry_run:
            print(f"[演练] 将移动目录: NER/ -> archive/NER/")
        else:
            try:
                target = archive_dir / 'NER'
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(ner_dir), str(target))
                print(f"✓ 已移动目录: NER/ -> archive/NER/")
            except Exception as e:
                print(f"✗ 移动失败: NER/ - {e}")
    
    # 移动converted_ner中未使用的数据集
    converted_ner_dir = datasets_path / 'converted_ner'
    if converted_ner_dir.exists():
        print(f"\n📁 归档未使用的转换数据集:")
        
        # 移动msra
        msra_dir = converted_ner_dir / 'msra'
        if msra_dir.exists():
            if dry_run:
                print(f"[演练] 将移动目录: converted_ner/msra/ -> archive/converted_ner/msra/")
            else:
                try:
                    target_parent = archive_dir / 'converted_ner'
                    target_parent.mkdir(exist_ok=True)
                    target = target_parent / 'msra'
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(msra_dir), str(target))
                    print(f"✓ 已移动目录: converted_ner/msra/ -> archive/converted_ner/msra/")
                except Exception as e:
                    print(f"✗ 移动失败: converted_ner/msra/ - {e}")
        
        # 移动peoplesdaily
        peoplesdaily_dir = converted_ner_dir / 'peoplesdaily'
        if peoplesdaily_dir.exists():
            if dry_run:
                print(f"[演练] 将移动目录: converted_ner/peoplesdaily/ -> archive/converted_ner/peoplesdaily/")
            else:
                try:
                    target_parent = archive_dir / 'converted_ner'
                    target_parent.mkdir(exist_ok=True)
                    target = target_parent / 'peoplesdaily'
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(peoplesdaily_dir), str(target))
                    print(f"✓ 已移动目录: converted_ner/peoplesdaily/ -> archive/converted_ner/peoplesdaily/")
                except Exception as e:
                    print(f"✗ 移动失败: converted_ner/peoplesdaily/ - {e}")
        
        # 移动weibo转换数据集（如果存在且与主weibo不同）
        weibo_converted = converted_ner_dir / 'weibo'
        if weibo_converted.exists():
            print(f"\n📁 处理转换的weibo数据集:")
            if dry_run:
                print(f"[演练] 将移动目录: converted_ner/weibo/ -> archive/converted_ner/weibo/")
                print(f"  (注意: 主weibo数据集保留在 datasets/weibo/)")
            else:
                try:
                    target_parent = archive_dir / 'converted_ner'
                    target_parent.mkdir(exist_ok=True)
                    target = target_parent / 'weibo'
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(weibo_converted), str(target))
                    print(f"✓ 已移动目录: converted_ner/weibo/ -> archive/converted_ner/weibo/")
                    print(f"  (主weibo数据集保留在 datasets/weibo/)")
                except Exception as e:
                    print(f"✗ 移动失败: converted_ner/weibo/ - {e}")
        
        # 移动配置文件
        config_file = converted_ner_dir / 'dataset_configs.py'
        if config_file.exists():
            if dry_run:
                print(f"[演练] 将移动文件: converted_ner/dataset_configs.py -> archive/")
            else:
                try:
                    target = archive_dir / 'dataset_configs.py'
                    shutil.move(str(config_file), str(target))
                    print(f"✓ 已移动文件: converted_ner/dataset_configs.py -> archive/")
                except Exception as e:
                    print(f"✗ 移动失败: dataset_configs.py - {e}")
        
        # 删除空的converted_ner目录
        if not dry_run and converted_ner_dir.exists():
            remaining = list(converted_ner_dir.iterdir())
            if len(remaining) == 0:
                try:
                    converted_ner_dir.rmdir()
                    print(f"✓ 已删除空目录: converted_ner/")
                except Exception as e:
                    print(f"⚠️  无法删除目录: converted_ner/ - {e}")
    
    # 创建数据集README
    readme_path = datasets_path / 'README.md'
    if not readme_path.exists():
        if dry_run:
            print(f"\n[演练] 将创建文件: datasets/README.md")
        else:
            readme_content = """# 数据集目录

本目录包含项目使用的数据集。

## 当前使用的数据集

### CLUENER2020
- 路径: `cluener/`
- 说明: 中文细粒度命名实体识别数据集
- 包含: train.json, dev.json, test.json, ent2id.json

### Weibo NER
- 路径: `weibo/`
- 说明: 微博命名实体识别数据集
- 包含: train.json, dev.json, test.json, ent2id.json

## 归档数据集

未使用的数据集已移动到 `archive/` 目录。

## 数据格式

所有数据集使用统一的JSON格式：

```json
{
  "text": "文本内容",
  "label": {
    "实体类型": {
      "实体文本": [[起始位置, 结束位置]]
    }
  }
}
```

## 添加自定义数据集

1. 创建数据集目录
2. 准备train.json, dev.json, test.json文件
3. 创建ent2id.json实体映射文件
4. 更新config.py中的数据集配置
"""
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print(f"\n✓ 已创建文件: datasets/README.md")
    
    print(f"\n{'='*60}")
    print(f"整理完成！")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='整理数据集目录')
    parser.add_argument('--dry-run', action='store_true', help='演练模式，不实际移动文件')
    parser.add_argument('--datasets-dir', default='datasets', help='数据集目录路径')
    
    args = parser.parse_args()
    
    if not args.dry_run:
        response = input("\n确认执行整理？这将移动未使用的数据集到archive目录。(y/N): ")
        if response.lower() != 'y':
            print("已取消整理操作")
            return
    
    organize_datasets(args.datasets_dir, args.dry_run)


if __name__ == '__main__':
    main()
