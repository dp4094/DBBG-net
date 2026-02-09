"""
文件清理工具类

用于识别和删除项目中的临时文件、缓存文件和无用文件。
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class FileCleanupPlan:
    """文件清理计划"""
    files_to_delete: List[str]  # 需要删除的文件
    dirs_to_delete: List[str]   # 需要删除的目录
    files_to_move: Dict[str, str]  # 需要移动的文件 {源: 目标}
    backup_required: bool = True  # 是否需要备份


class FileCleanup:
    """文件清理工具类"""
    
    # 临时文件模式
    TEMP_FILE_PATTERNS = [
        r'^temp_.*\.py$',
        r'^_[^_].*\.py$',  # 以单个下划线开头的py文件，但不包括__init__.py等
        r'.*\.tmp$',
        r'.*\.bak$',
        r'^新建文本文档.*\.txt$',
        r'^out\.txt$',
        r'^1\.md$',
    ]
    
    # 压缩包模式
    ARCHIVE_PATTERNS = [
        r'.*\.zip$',
        r'.*\.tar\.gz$',
        r'.*\.rar$',
    ]
    
    # Python缓存
    PYTHON_CACHE_PATTERNS = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
    ]
    
    # IDE配置目录
    IDE_DIRS = ['.idea', '.vscode']
    
    # 需要排除的目录
    EXCLUDE_DIRS = ['venv', 'env', '.git', '.kiro', 'node_modules']
    
    def __init__(self, project_root: str = '.'):
        """
        初始化文件清理工具
        
        Args:
            project_root: 项目根目录路径
        """
        self.project_root = Path(project_root).resolve()
        
    def _should_exclude_path(self, path: Path) -> bool:
        """检查路径是否应该被排除"""
        parts = path.relative_to(self.project_root).parts
        return any(exclude_dir in parts for exclude_dir in self.EXCLUDE_DIRS)
    
    def identify_temp_files(self) -> List[str]:
        """
        识别所有临时文件
        
        Returns:
            临时文件路径列表
        """
        temp_files = []
        
        for pattern in self.TEMP_FILE_PATTERNS:
            regex = re.compile(pattern)
            for file_path in self.project_root.rglob('*'):
                if file_path.is_file() and regex.match(file_path.name):
                    if not self._should_exclude_path(file_path):
                        temp_files.append(str(file_path.relative_to(self.project_root)))
        
        return temp_files
    
    def identify_archive_files(self) -> List[str]:
        """
        识别所有归档文件（压缩包）
        
        Returns:
            归档文件路径列表
        """
        archive_files = []
        
        for pattern in self.ARCHIVE_PATTERNS:
            regex = re.compile(pattern)
            for file_path in self.project_root.rglob('*'):
                if file_path.is_file() and regex.match(file_path.name):
                    if not self._should_exclude_path(file_path):
                        archive_files.append(str(file_path.relative_to(self.project_root)))
        
        return archive_files
    
    def identify_python_cache(self) -> List[str]:
        """
        识别所有Python缓存文件和目录
        
        Returns:
            缓存文件/目录路径列表
        """
        cache_items = []
        
        # 查找__pycache__目录
        for cache_dir in self.project_root.rglob('__pycache__'):
            if cache_dir.is_dir() and not self._should_exclude_path(cache_dir):
                cache_items.append(str(cache_dir.relative_to(self.project_root)))
        
        # 查找.pyc和.pyo文件
        for pyc_file in self.project_root.rglob('*.pyc'):
            if pyc_file.is_file() and not self._should_exclude_path(pyc_file):
                cache_items.append(str(pyc_file.relative_to(self.project_root)))
        
        for pyo_file in self.project_root.rglob('*.pyo'):
            if pyo_file.is_file() and not self._should_exclude_path(pyo_file):
                cache_items.append(str(pyo_file.relative_to(self.project_root)))
        
        return cache_items

    def identify_ide_configs(self) -> List[str]:
        """
        识别IDE配置目录
        
        Returns:
            IDE配置目录路径列表
        """
        ide_dirs = []
        
        for ide_dir_name in self.IDE_DIRS:
            ide_path = self.project_root / ide_dir_name
            if ide_path.exists() and ide_path.is_dir():
                ide_dirs.append(str(ide_path.relative_to(self.project_root)))
        
        return ide_dirs
    
    def create_cleanup_plan(self, 
                           clean_temp: bool = True,
                           clean_cache: bool = True,
                           clean_archives: bool = True,
                           clean_ide: bool = True) -> FileCleanupPlan:
        """
        创建文件清理计划
        
        Args:
            clean_temp: 是否清理临时文件
            clean_cache: 是否清理Python缓存
            clean_archives: 是否清理压缩包
            clean_ide: 是否清理IDE配置
            
        Returns:
            文件清理计划
        """
        files_to_delete = []
        dirs_to_delete = []
        
        if clean_temp:
            files_to_delete.extend(self.identify_temp_files())
        
        if clean_archives:
            files_to_delete.extend(self.identify_archive_files())
        
        if clean_cache:
            cache_items = self.identify_python_cache()
            for item in cache_items:
                item_path = self.project_root / item
                if item_path.is_dir():
                    dirs_to_delete.append(item)
                else:
                    files_to_delete.append(item)
        
        if clean_ide:
            dirs_to_delete.extend(self.identify_ide_configs())
        
        return FileCleanupPlan(
            files_to_delete=sorted(set(files_to_delete)),
            dirs_to_delete=sorted(set(dirs_to_delete)),
            files_to_move={},
            backup_required=True
        )
    
    def remove_file(self, file_path: str, dry_run: bool = False) -> bool:
        """
        安全删除文件
        
        Args:
            file_path: 相对于项目根目录的文件路径
            dry_run: 是否为演练模式（不实际删除）
            
        Returns:
            是否成功删除
        """
        full_path = self.project_root / file_path
        
        if not full_path.exists():
            print(f"⚠️  文件不存在: {file_path}")
            return False
        
        if not full_path.is_file():
            print(f"⚠️  不是文件: {file_path}")
            return False
        
        if dry_run:
            print(f"[演练] 将删除文件: {file_path}")
            return True
        
        try:
            full_path.unlink()
            print(f"✓ 已删除文件: {file_path}")
            return True
        except Exception as e:
            print(f"✗ 删除文件失败: {file_path} - {e}")
            return False
    
    def remove_directory(self, dir_path: str, dry_run: bool = False) -> bool:
        """
        安全删除目录
        
        Args:
            dir_path: 相对于项目根目录的目录路径
            dry_run: 是否为演练模式（不实际删除）
            
        Returns:
            是否成功删除
        """
        full_path = self.project_root / dir_path
        
        if not full_path.exists():
            print(f"⚠️  目录不存在: {dir_path}")
            return False
        
        if not full_path.is_dir():
            print(f"⚠️  不是目录: {dir_path}")
            return False
        
        if dry_run:
            print(f"[演练] 将删除目录: {dir_path}")
            return True
        
        try:
            shutil.rmtree(full_path)
            print(f"✓ 已删除目录: {dir_path}")
            return True
        except Exception as e:
            print(f"✗ 删除目录失败: {dir_path} - {e}")
            return False
    
    def execute_cleanup_plan(self, plan: FileCleanupPlan, dry_run: bool = False) -> Dict[str, int]:
        """
        执行清理计划
        
        Args:
            plan: 文件清理计划
            dry_run: 是否为演练模式（不实际删除）
            
        Returns:
            清理统计信息 {'files_deleted': n, 'dirs_deleted': m, 'failed': k}
        """
        stats = {
            'files_deleted': 0,
            'dirs_deleted': 0,
            'failed': 0
        }
        
        print(f"\n{'='*60}")
        print(f"{'[演练模式] ' if dry_run else ''}开始执行文件清理")
        print(f"{'='*60}\n")
        
        # 删除文件
        if plan.files_to_delete:
            print(f"📄 删除文件 ({len(plan.files_to_delete)} 个):")
            for file_path in plan.files_to_delete:
                if self.remove_file(file_path, dry_run):
                    stats['files_deleted'] += 1
                else:
                    stats['failed'] += 1
        
        # 删除目录
        if plan.dirs_to_delete:
            print(f"\n📁 删除目录 ({len(plan.dirs_to_delete)} 个):")
            for dir_path in plan.dirs_to_delete:
                if self.remove_directory(dir_path, dry_run):
                    stats['dirs_deleted'] += 1
                else:
                    stats['failed'] += 1
        
        print(f"\n{'='*60}")
        print(f"清理完成！")
        print(f"  - 文件删除: {stats['files_deleted']}")
        print(f"  - 目录删除: {stats['dirs_deleted']}")
        print(f"  - 失败: {stats['failed']}")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """主函数 - 命令行使用示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='项目文件清理工具')
    parser.add_argument('--dry-run', action='store_true', help='演练模式，不实际删除文件')
    parser.add_argument('--no-temp', action='store_true', help='不清理临时文件')
    parser.add_argument('--no-cache', action='store_true', help='不清理Python缓存')
    parser.add_argument('--no-archives', action='store_true', help='不清理压缩包')
    parser.add_argument('--no-ide', action='store_true', help='不清理IDE配置')
    parser.add_argument('--project-root', default='.', help='项目根目录路径')
    
    args = parser.parse_args()
    
    # 创建清理工具
    cleanup = FileCleanup(args.project_root)
    
    # 创建清理计划
    plan = cleanup.create_cleanup_plan(
        clean_temp=not args.no_temp,
        clean_cache=not args.no_cache,
        clean_archives=not args.no_archives,
        clean_ide=not args.no_ide
    )
    
    # 显示清理计划
    print("\n清理计划:")
    print(f"  - 临时文件: {len([f for f in plan.files_to_delete if any(re.match(p, Path(f).name) for p in FileCleanup.TEMP_FILE_PATTERNS)])} 个")
    print(f"  - 压缩包: {len([f for f in plan.files_to_delete if any(re.match(p, Path(f).name) for p in FileCleanup.ARCHIVE_PATTERNS)])} 个")
    print(f"  - Python缓存: {len(plan.dirs_to_delete) + len([f for f in plan.files_to_delete if f.endswith(('.pyc', '.pyo'))])} 项")
    print(f"  - IDE配置: {len([d for d in plan.dirs_to_delete if Path(d).name in FileCleanup.IDE_DIRS])} 个")
    
    # 确认执行
    if not args.dry_run:
        response = input("\n确认执行清理？(y/N): ")
        if response.lower() != 'y':
            print("已取消清理操作")
            return
    
    # 执行清理
    cleanup.execute_cleanup_plan(plan, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
