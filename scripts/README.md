# 脚本目录

本目录包含项目的辅助脚本和工具。

## 脚本列表

### 数据处理
- `batch_convert_datasets.py` - 批量转换数据集格式

### 模型转换
- `convert_structure_to_onnx.py` - 将模型转换为ONNX格式

### 项目维护
- `file_cleanup.py` - 文件清理工具
- `organize_outputs.py` - 整理训练输出目录

## 使用方法

每个脚本都可以独立运行，使用 `--help` 查看帮助：

```bash
python scripts/file_cleanup.py --help
```

## 注意事项

- 某些脚本可能会修改或删除文件，请谨慎使用
- 建议先使用 `--dry-run` 参数进行演练
- 详细说明请查看各脚本的文档字符串
