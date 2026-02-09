# Implementation Plan: GlobalPointer项目整理规范

## Overview

本实施计划将GlobalPointer中文命名实体识别项目的整理工作分解为一系列可执行的任务。整理过程包括文件清理、项目结构重组、文档生成、配置更新等步骤，最终使项目达到开源发布的质量标准。

## Tasks

- [x] 1. 创建项目备份和准备工作
  - 创建完整的项目备份
  - 初始化Git仓库（如果尚未初始化）
  - 创建新的工作分支用于整理
  - _Requirements: 10.1, 10.4_

- [x] 2. 实现文件清理模块
  - [x] 2.1 创建文件清理工具类
    - 实现FileCleanup类，包含文件识别和删除方法
    - 添加临时文件模式匹配逻辑
    - 实现安全删除功能（带确认和备份）
    - _Requirements: 1.1, 1.4, 1.7, 1.8_

  - [ ]* 2.2 编写文件清理属性测试
    - **Property 1: 临时文件识别完整性**
    - **Property 2: Python缓存清理完整性**
    - **Property 4: 归档文件识别完整性**
    - **Property 5: 临时文本文件模式匹配**
    - **Validates: Requirements 1.1, 1.4, 1.7, 1.8**

  - [x] 2.3 执行文件清理操作
    - 删除临时测试文件（temp_*.py, _*.py等）
    - 清理Python缓存（__pycache__, *.pyc）
    - 删除压缩包和临时文本文件
    - 清理IDE配置目录（.idea, .vscode）
    - _Requirements: 1.1, 1.4, 1.5, 1.7, 1.8_

  - [x] 2.4 整理outputs目录
    - 实现模型检查点筛选逻辑（保留最佳N个）
    - 清理旧的训练日志
    - 组织输出文件结构
    - _Requirements: 1.6_

  - [ ]* 2.5 编写输出整理属性测试
    - **Property 3: 输出文件保留策略正确性**
    - **Validates: Requirements 1.6**

- [x] 3. Checkpoint - 验证文件清理结果
  - 确认所有临时文件已删除
  - 验证重要文件未被误删
  - 询问用户是否有问题

- [x] 4. 实现项目结构优化模块
  - [x] 4.1 创建标准目录结构
    - 创建docs/、scripts/、examples/、tests/目录
    - 为每个目录添加README.md说明文件
    - 创建.gitkeep文件保持空目录
    - _Requirements: 4.5, 4.6, 4.7_

  - [x] 4.2 重组源代码文件（可选）
    - 评估是否需要创建src/目录
    - 如果重组，更新所有import语句
    - 确保代码仍可正常运行
    - _Requirements: 4.1_

  - [x] 4.3 整理数据集目录
    - 统一数据集目录结构
    - 为每个数据集添加README说明
    - 删除或归档未使用的数据集（msra, peoplesdaily）
    - _Requirements: 4.2_

  - [ ]* 4.4 编写结构优化属性测试
    - **Property 14: 目录结构规范性**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [x] 5. 更新配置文件
  - [x] 5.1 修正main.py为正确的训练入口
    - 检查当前main.py内容
    - 如果是pywebview代码，用train.py的内容替换
    - 或者创建新的main.py作为统一入口
    - _Requirements: 5.1_

  - [x] 5.2 优化config.py
    - 为所有配置参数添加详细注释
    - 添加参数类型说明和默认值
    - 使用环境变量替代硬编码路径
    - 添加配置验证函数
    - _Requirements: 5.2, 5.5_

  - [ ]* 5.3 编写配置参数属性测试
    - **Property 8: 配置参数注释完整性**
    - **Property 10: 路径格式规范性**
    - **Validates: Requirements 5.2, 5.5**

  - [x] 5.4 创建.gitignore文件
    - 添加Python标准忽略规则
    - 添加IDE配置忽略
    - 添加大文件目录忽略（pretrained_models, outputs）
    - 添加数据集文件忽略
    - _Requirements: 5.3_

  - [ ]* 5.5 编写.gitignore属性测试
    - **Property 9: .gitignore规则完整性**
    - **Validates: Requirements 5.3**

  - [x] 5.6 创建.env.example模板
    - 定义所有环境变量
    - 提供示例值和说明
    - _Requirements: 5.4_

- [ ] 6. 生成完整的requirements.txt
  - [x] 6.1 分析项目依赖
    - 扫描所有Python文件的import语句
    - 识别第三方包
    - 确定每个包的版本号
    - _Requirements: 2.1_

  - [x] 6.2 编写requirements.txt
    - 按类别组织依赖（核心、可选、开发）
    - 为每个依赖添加注释说明用途
    - 指定版本号或版本范围
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ]* 6.3 编写依赖管理属性测试
    - **Property 6: 依赖列表完整性**
    - **Property 7: 依赖分类正确性**
    - **Validates: Requirements 2.1, 2.3**

- [ ] 7. Checkpoint - 验证配置更新
  - 测试新的配置文件是否正常工作
  - 验证环境变量配置
  - 询问用户是否有问题

- [ ] 8. 生成中文README.md
  - [ ] 8.1 编写项目简介部分
    - 项目背景和目标
    - GlobalPointer模型原理
    - 主要特性列表
    - _Requirements: 3.2_

  - [ ] 8.2 编写性能表现部分
    - 各数据集的F1分数表格
    - 训练曲线图（如果有）
    - 与其他方法的对比
    - _Requirements: 3.2, 6.1_

  - [ ] 8.3 编写快速开始部分
    - 环境要求说明
    - 安装步骤（conda环境、依赖安装）
    - 快速训练示例
    - 快速预测示例
    - _Requirements: 2.4, 2.5, 3.3_

  - [ ] 8.4 编写数据集部分
    - 支持的数据集列表
    - 数据格式说明（JSON格式）
    - 数据下载链接
    - 自定义数据集准备指南
    - _Requirements: 3.4, 8.1, 8.2, 8.5_

  - [ ] 8.5 编写训练和评估部分
    - 训练命令和参数说明
    - 评估命令
    - 超参数调优建议
    - _Requirements: 3.4_

  - [ ] 8.6 编写项目结构部分
    - 目录树展示
    - 核心文件说明
    - _Requirements: 3.1_

  - [ ] 8.7 编写贡献和许可证部分
    - 贡献指南链接
    - 许可证信息
    - 引用信息
    - _Requirements: 3.6, 3.7_

  - [ ]* 8.8 编写README文档属性测试
    - **Property 11: 文档结构完整性**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

- [ ] 9. 生成英文README_EN.md
  - [ ] 9.1 翻译中文README内容
    - 翻译所有章节为英文
    - 保持格式和结构一致
    - _Requirements: 3.1_

- [ ] 10. 创建详细文档
  - [ ] 10.1 创建docs/installation.md
    - 详细的环境配置步骤
    - 预训练模型下载指南
    - 常见安装问题解决
    - _Requirements: 3.3_

  - [ ] 10.2 创建docs/usage.md
    - 详细的训练流程
    - 配置参数详解
    - 评估和预测使用方法
    - _Requirements: 3.4_

  - [ ] 10.3 创建docs/api.md
    - 主要类和函数的API文档
    - 使用示例
    - _Requirements: 3.5_

  - [ ] 10.4 创建docs/performance.md
    - 详细的性能报告
    - 各数据集的P/R/F1指标
    - 各实体类型的性能分析
    - 训练环境和时间信息
    - 不同配置的性能对比
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 10.5 编写性能报告属性测试
    - **Property 12: 性能报告完整性**
    - **Validates: Requirements 6.1, 6.3, 6.4, 6.5**

  - [ ] 10.6 创建docs/dataset.md
    - JSON数据格式详细说明
    - 字段含义解释
    - 数据集统计信息
    - 数据准备脚本使用
    - _Requirements: 8.1, 8.3, 8.4_

  - [ ]* 10.7 编写数据集文档属性测试
    - **Property 13: 数据集文档完整性**
    - **Validates: Requirements 8.1**

  - [ ] 10.8 创建docs/deployment.md
    - ONNX模型导出指南
    - 推理API使用
    - 批量处理示例
    - Web服务集成示例
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 11. 创建示例代码
  - [ ] 11.1 创建examples/quick_start.py
    - 最简单的训练和预测示例
    - 详细注释
    - _Requirements: 3.4_

  - [ ] 11.2 创建examples/custom_dataset.py
    - 自定义数据集准备示例
    - 数据格式转换示例
    - _Requirements: 8.5_

  - [ ] 11.3 创建examples/inference_demo.py
    - 模型加载和推理示例
    - 批量处理示例
    - _Requirements: 9.2, 9.4_

  - [ ]* 11.4 编写批量处理功能测试
    - **Property 18: 批量处理功能正确性**
    - **Validates: Requirements 9.4**

- [ ] 12. 创建辅助脚本
  - [ ] 12.1 创建scripts/download_models.sh
    - 自动下载预训练模型脚本
    - 支持多个模型选择
    - _Requirements: 3.3_

  - [ ] 12.2 整理现有脚本
    - 移动batch_convert_datasets.py到scripts/
    - 移动convert_structure_to_onnx.py到scripts/
    - 添加脚本使用说明
    - _Requirements: 8.3, 9.1_

  - [ ]* 12.3 编写ONNX导出功能测试
    - **Property 17: ONNX导出功能正确性**
    - **Validates: Requirements 9.1**

- [ ] 13. Checkpoint - 验证文档和示例
  - 检查所有文档链接是否正确
  - 运行示例代码验证可用性
  - 询问用户是否有问题

- [ ] 14. 改进代码质量
  - [ ] 14.1 添加类型注解
    - 为关键函数添加类型提示
    - 使用typing模块
    - _Requirements: 7.2_

  - [ ] 14.2 完善文档字符串
    - 为所有公共函数添加docstring
    - 使用Google或NumPy风格
    - _Requirements: 7.3_

  - [ ] 14.3 统一日志格式
    - 将所有print语句改为logging
    - 使用统一的日志格式
    - _Requirements: 7.5_

  - [ ] 14.4 代码风格检查
    - 运行black格式化代码
    - 运行flake8检查代码风格
    - 修复所有PEP 8违规
    - _Requirements: 7.1_

  - [ ]* 14.5 编写代码质量属性测试
    - **Property 15: 代码规范性**
    - **Property 16: 日志格式一致性**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.5**

- [ ] 15. 创建贡献指南和许可证
  - [ ] 15.1 创建CONTRIBUTING.md
    - 代码规范说明
    - 提交流程
    - Pull Request要求
    - 测试要求
    - _Requirements: 3.6_

  - [ ] 15.2 创建LICENSE文件
    - 选择合适的开源许可证（MIT或Apache 2.0）
    - 添加许可证文本
    - _Requirements: 3.7_

  - [ ] 15.3 创建CHANGELOG.md
    - 添加版本历史
    - 记录主要变更
    - _Requirements: 10.3_

  - [ ]* 15.4 编写版本管理属性测试
    - **Property 19: 版本号格式规范性**
    - **Property 20: 变更日志完整性**
    - **Validates: Requirements 10.2, 10.3**

- [ ] 16. 更新AGENTS.md为项目规范文档
  - [ ] 16.1 编写项目结构说明
    - 目录组织规范
    - 模块职责说明
    - _Requirements: 1.2_

  - [ ] 16.2 编写开发命令说明
    - 环境配置命令
    - 训练和评估命令
    - 测试命令
    - _Requirements: 2.4, 2.5_

  - [ ] 16.3 编写代码规范说明
    - 命名约定
    - 文档字符串规范
    - 提交规范
    - _Requirements: 7.1, 7.2, 7.3_

- [ ] 17. 最终检查和测试
  - [ ] 17.1 运行完整测试套件
    - 运行所有单元测试
    - 运行所有属性测试
    - 确保测试覆盖率>80%
    - _Requirements: 所有测试相关需求_

  - [ ] 17.2 验证安装流程
    - 在干净环境中测试安装
    - 验证requirements.txt的完整性
    - 测试快速开始示例
    - _Requirements: 2.1, 3.3_

  - [ ] 17.3 验证文档完整性
    - 检查所有文档链接
    - 验证代码示例可运行
    - 检查拼写和格式
    - _Requirements: 3.1-3.7_

  - [ ] 17.4 创建版本标签
    - 创建v1.0.0版本标签
    - 推送到远程仓库
    - _Requirements: 10.2_

- [ ] 18. Final Checkpoint - 项目整理完成
  - 确认所有任务已完成
  - 验证项目可以正常运行
  - 准备发布到GitHub
  - 询问用户是否满意

## Notes

- 标记为`*`的任务是可选的测试任务，可以跳过以加快进度
- 每个任务都引用了相关的需求编号，便于追溯
- Checkpoint任务用于阶段性验证和用户确认
- 属性测试使用hypothesis库，每个测试至少运行100次
- 建议按顺序执行任务，确保依赖关系正确
- 文件操作前建议先备份项目
