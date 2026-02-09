# 测试目录

本目录包含项目的单元测试和集成测试。

## 测试框架

本项目使用 `pytest` 作为测试框架。

## 运行测试

运行所有测试：
```bash
pytest tests/
```

运行特定测试文件：
```bash
pytest tests/test_model.py
```

查看测试覆盖率：
```bash
pytest --cov=. tests/
```

## 测试文件

- `test_model.py` - 模型相关测试
- `test_data_utils.py` - 数据处理测试
- `test_utils.py` - 工具函数测试

## 编写测试

测试文件命名规范：`test_*.py`
测试函数命名规范：`test_*`

示例：
```python
def test_model_forward():
    """测试模型前向传播"""
    # 测试代码
    pass
```
