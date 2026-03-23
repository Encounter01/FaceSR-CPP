# FaceSR_CPP 项目文档索引

## 核心项目文档

### 1. 项目概览
- **README.md** - 项目基本信息和使用说明

### 2. 快速开始
- **QUICKSTART.md** - 新用户快速上手指南
  - 环境配置
  - 编译构建
  - 基本使用

### 3. 评估和验证
- **EVALUATION.md** - 项目评估方法和结果
  - 评估流程
  - 性能指标
  - 如何复现结果

---

## 毕设答辩相关文档（核心）

### 毕设完成度评估
- **results/eval_reports/THESIS_ASSESSMENT.md** ⭐
  - 本科毕设完成度评估
  - 毕设要求完成情况
  - 预期答辩评分
  - 答辩关键问题准备

### 项目总结报告
- **results/eval_reports/FINAL_SUMMARY.md** ⭐
  - 完整项目执行总结
  - 生成文件清单
  - 验证检查表

### 模型选择指南
- **results/eval_reports/MODEL_SELECTION_GUIDE.md** ⭐
  - 模型性能对比
  - 部署建议
  - 监控指标

### 根因分析报告
- **results/eval_reports/ROOT_CAUSE_ANALYSIS.md**
  - Best/Epoch190 异常分析
  - 调查过程和发现
  - 补救方案

---

## 技术设计文档

### 系统架构
- **docs/design/ARCHITECTURE.md**
  - 整体系统设计
  - 模块构成
  - 数据流

### 算法设计
- **docs/design/ALGORITHM.md**
  - RRDB 网络架构
  - 损失函数设计
  - 训练策略

### API 文档
- **docs/API.md**
  - 关键类和函数说明
  - 接口定义

---

## 使用教程

### 环境配置
- **docs/tutorial/SETUP.md**
  - 依赖库安装
  - 编译步骤
  - 常见问题

### 训练指南
- **docs/tutorial/TRAINING.md**
  - 数据准备
  - 训练命令
  - 参数说明

### 推理指南
- **docs/tutorial/INFERENCE.md**
  - 推理使用
  - 批量处理
  - 性能优化

---

## 评估数据和可视化

### 关键数据文件
- **results/eval_reports/full_metrics_report.csv**
  - 3000 张测试图像的详细指标
  - 包含 PSNR, SSIM, LPIPS, 亮度, 边缘密度

- **results/eval_reports/stratified_analysis_corrected.csv**
  - 按亮度和边缘密度的分层统计
  - 各类别的性能对比

### 可视化文件
- **results/eval_reports/checkpoint_comparison_bar.png**
  - 性能对比柱状图

- **results/eval_reports/panels/**
  - 对比面板 (60×4 = 240 张)
  - 展示最优/最差/随机样本

---

## 项目源代码

### C++ 代码
```
include/              - C++ 头文件
  ├── models/        - 神经网络定义
  ├── utils/         - 工具函数
  ├── gui/           - GUI 组件
  └── ...

src/                  - C++ 源文件
  ├── models/        - 网络实现
  ├── utils/         - 工具实现
  ├── gui/           - GUI 实现
  ├── trainer.cpp    - 训练逻辑
  ├── inference.cpp  - 推理逻辑
  └── main_*.cpp     - 程序入口
```

### 脚本
```
scripts/
  ├── evaluation/    - 评估脚本
  │   ├── evaluate_model.py
  │   ├── evaluate_model_final.py
  │   └── ...
  └── utils/         - 工具脚本
      ├── gen_lr.py
      └── extract_and_prepare.py
```

---

## 配置和构建

### 构建配置
- **CMakeLists.txt** - CMake 项目配置
- **CMakePresets.json** - CMake 预设

### 项目配置
- **config/train_config.ini** - 训练参数配置
- **config/ui_config.ini** - UI 样式配置

---

## 许可和引用

- **LICENSE** - CC BY-NC-SA 4.0 许可证
- **README.md** - 项目引用信息

---

## 数据目录（不上传）

```
data/
  ├── train/        - 训练数据
  ├── val/          - 验证数据
  └── test/         - 测试数据 (3000 张 CelebA)

checkpoints/        - 模型权重（精选）
  ├── generator_latest.pt
  ├── generator_best.pt
  └── generator_epoch190.pt
```

---

## 推荐阅读顺序

### 对于答辩评委
1. **README.md** - 项目概览 (5 分钟)
2. **THESIS_ASSESSMENT.md** - 毕设评估 (15 分钟)
3. **FINAL_SUMMARY.md** - 项目总结 (10 分钟)
4. 性能对比图和面板 (5 分钟)

### 对于想深入了解的人
1. **ARCHITECTURE.md** - 系统设计
2. **ALGORITHM.md** - 算法细节
3. **EVALUATION.md** - 评估方法
4. **ROOT_CAUSE_ANALYSIS.md** - 技术深度

### 对于想复现的人
1. **QUICKSTART.md** - 快速开始
2. **SETUP.md** - 环境配置
3. **TRAINING.md** - 训练指南
4. **INFERENCE.md** - 推理使用

---

最后更新: 2026-03-23
