# FaceSR_CPP 项目结构说明

## 完整项目结构

```
FaceSR_CPP/
│
├── 📄 README.md                          项目概览和使用说明
├── 📄 DOCS.md                            文档索引（导航）
├── 📄 QUICKSTART.md                      快速开始指南
├── 📄 EVALUATION.md                      评估方法说明
├── 📄 LICENSE                            许可证（CC BY-NC-SA 4.0）
├── 📄 CMakeLists.txt                     CMake 项目配置
├── 📄 CMakePresets.json                  CMake 预设配置
│
│
├── 📁 include/                           C++ 头文件目录
│   ├── common/
│   │   ├── config.h                      配置管理
│   │   ├── config_parser.h               配置解析
│   │   ├── logger.h                      日志记录
│   │   └── random.h                      随机数生成
│   │
│   ├── models/
│   │   ├── generator.h                   生成器网络定义
│   │   ├── discriminator.h               判别器网络定义
│   │   └── losses.h                      损失函数定义
│   │
│   ├── utils/
│   │   ├── dataset.h                     数据集加载
│   │   ├── image_utils.h                 图像处理工具
│   │   └── metrics.h                     性能指标计算
│   │
│   ├── gui/
│   │   ├── main_window.h                 主窗口
│   │   ├── style_config.h                样式配置
│   │   ├── background_widget.h           背景渲染
│   │   └── frosted_widget.h              毛玻璃效果
│   │
│   ├── trainer.h                         训练器定义
│   └── inference.h                       推理器定义
│
│
├── 📁 src/                               C++ 源文件目录
│   ├── models/
│   │   ├── generator.cpp                 生成器实现
│   │   ├── discriminator.cpp             判别器实现
│   │   └── losses.cpp                    损失函数实现
│   │
│   ├── utils/
│   │   ├── dataset.cpp                   数据加载实现
│   │   ├── image_utils.cpp               图像处理实现
│   │   └── metrics.cpp                   指标计算实现
│   │
│   ├── gui/
│   │   ├── main_window.cpp               主窗口实现
│   │   ├── style_config.cpp              样式实现
│   │   ├── background_widget.cpp         背景实现
│   │   └── frosted_widget.cpp            毛玻璃实现
│   │
│   ├── trainer.cpp                       训练器实现
│   ├── inference.cpp                     推理器实现
│   │
│   ├── main_train.cpp                    训练程序入口
│   ├── main_test.cpp                     测试/推理程序入口
│   └── main_gui.cpp                      GUI 程序入口
│
│
├── 📁 config/                            配置文件目录
│   ├── train_config.ini                  训练参数配置
│   ├── train_config_sharper.ini          更锐利的训练配置
│   ├── ui_config.ini                     UI 样式配置
│   ├── finetune_phase_a.ini              微调阶段 A
│   ├── finetune_phase_b.ini              微调阶段 B
│   └── finetune_phase_c.ini              微调阶段 C
│
│
├── 📁 data/                              数据目录（不上传）
│   ├── train/
│   │   ├── HR/                           高分训练图像
│   │   └── LR/                           低分训练图像
│   │
│   ├── val/
│   │   ├── HR/                           高分验证图像
│   │   └── LR/                           低分验证图像
│   │
│   └── test/
│       ├── HR/                           高分测试图像 (3000 张)
│       └── LR/                           低分测试图像 (3000 张)
│
│
├── 📁 checkpoints/                       模型权重目录（精选）
│   ├── generator_latest.pt               最新生成器模型（推荐）
│   ├── generator_best.pt                 最佳生成器模型（已弃用）
│   └── generator_epoch190.pt             第 190 轮生成器模型
│
│
├── 📁 results/                           评估结果目录
│   ├── eval_reports/
│   │   ├── THESIS_ASSESSMENT.md          ⭐ 毕设完成度评估
│   │   ├── FINAL_SUMMARY.md              ⭐ 项目最终总结
│   │   ├── MODEL_SELECTION_GUIDE.md      ⭐ 模型选择指南
│   │   ├── ROOT_CAUSE_ANALYSIS.md        根因分析报告
│   │   │
│   │   ├── full_metrics_report.csv       3000 张逐图详细指标
│   │   ├── stratified_analysis_corrected.csv  分层分析结果
│   │   ├── checkpoint_comparison_bar.png 性能对比柱状图
│   │   │
│   │   └── panels/                       对比面板图（240 张）
│   │       ├── latest/                   Latest 模型面板 (60 张)
│   │       ├── best/                     Best 模型面板 (60 张)
│   │       ├── epoch190/                 Epoch190 模型面板 (60 张)
│   │       └── bicubic/                  Bicubic 基线面板 (60 张)
│   │
│   └── analysis/                         分析结果（预留）
│
│
├── 📁 scripts/                           辅助脚本目录
│   │
│   ├── evaluation/                       评估脚本
│   │   ├── evaluate_model.py             主评估脚本（12 步完整流程）
│   │   ├── evaluate_model_final.py       数据质量验证
│   │   ├── diagnose_sr_issue.py          SR 图像诊断
│   │   ├── analyze_checkpoint_quality.py Checkpoint 质量分析
│   │   ├── training_decision_analysis.py 训练决策分析
│   │   ├── generate_root_cause_analysis.py 根因分析生成
│   │   └── generate_final_summary.py     总结报告生成
│   │
│   └── utils/                            工具脚本
│       ├── gen_lr.py                     生成低分图像
│       └── extract_and_prepare.py        数据提取和准备
│
│
├── 📁 docs/                              项目技术文档
│   │
│   ├── API.md                            API 文档（C++ 接口）
│   │
│   ├── design/                           设计文档
│   │   ├── ARCHITECTURE.md               系统架构设计
│   │   ├── ALGORITHM.md                  算法设计说明
│   │   └── PERFORMANCE.md                性能优化说明
│   │
│   ├── api/                              API 详细文档
│   │   ├── generator.md                  生成器 API
│   │   ├── trainer.md                    训练器 API
│   │   └── inference.md                  推理器 API
│   │
│   └── tutorial/                         使用教程
│       ├── SETUP.md                      环境配置教程
│       ├── TRAINING.md                   训练指南
│       ├── INFERENCE.md                  推理指南
│       └── GUI.md                        GUI 使用指南
│
│
├── 📁 thesis/                            论文相关文件
│   ├── figures/                          论文图表
│   │   ├── architecture.pdf              系统架构图
│   │   ├── results_comparison.pdf        性能对比图
│   │   └── ...
│   │
│   ├── tables/                           论文表格
│   │   ├── performance_metrics.xlsx      性能指标表
│   │   ├── ablation_study.xlsx           消融实验表
│   │   └── ...
│   │
│   └── references/                       参考文献
│       └── references.bib                BibTeX 格式参考文献
│
│
├── 📁 build/                             构建目录（不上传）
│   ├── bin/Release/                      编译输出的可执行文件
│   │   ├── facesr_train.exe              训练程序
│   │   ├── facesr_test.exe               测试/推理程序
│   │   └── facesr_gui_app.exe            GUI 应用
│   │
│   └── ... (CMake 生成的临时文件)
│
│
└── 📁 .git/                              Git 版本控制（不上传）
    └── ... (版本历史)
```

---

## 目录说明

### 源代码目录 (include + src)

- **models/** - 神经网络定义和实现
  - generator.h/cpp: RRDB 生成器 (23 个 RRDB 块)
  - discriminator.h/cpp: VGG 风格判别器
  - losses.h/cpp: 多项损失函数 (L1 + VGG + GAN)

- **utils/** - 工具函数库
  - dataset.h/cpp: 多线程数据加载
  - image_utils.h/cpp: 图像处理 (缩放、正则化等)
  - metrics.h/cpp: 性能指标 (PSNR, SSIM, LPIPS)

- **gui/** - GUI 组件 (Qt 6)
  - main_window.h/cpp: 主应用窗口
  - frosted_widget.h/cpp: 毛玻璃效果
  - style_config.h/cpp: 主题和样式

- **common/** - 公共工具
  - config_parser.h/cpp: INI 配置文件解析
  - logger.h/cpp: 日志记录
  - random.h: 随机数生成

### 配置目录 (config/)

- **train_config.ini** - 训练参数配置（数据路径、超参数、损失权重）
- **ui_config.ini** - GUI 样式配置（背景、主题、毛玻璃）

### 数据目录 (data/)

**注意**: 数据目录不上传到 GitHub（过大），通过脚本生成或单独下载

- **train/** - 训练集 (HR 和 LR 成对)
- **val/** - 验证集
- **test/** - 测试集 (3000 张 CelebA 人脸)

### 评估结果 (results/)

**关键文件** - 毕设答辩必读:
- THESIS_ASSESSMENT.md - 毕设完成度评估
- FINAL_SUMMARY.md - 项目总结
- MODEL_SELECTION_GUIDE.md - 模型选择建议

**数据文件**:
- full_metrics_report.csv - 3000 张图像的详细指标
- stratified_analysis_corrected.csv - 分层统计分析

**可视化**:
- checkpoint_comparison_bar.png - 性能对比柱状图
- panels/ - 对比面板图（最优/最差/随机样本）

### 脚本目录 (scripts/)

**evaluation/** - 评估脚本（可独立运行）
- evaluate_model.py - 主评估流程（12 步）
- evaluate_model_final.py - 数据质量检验
- diagnose_sr_issue.py - SR 图像诊断
- ...其他诊断和分析脚本

**utils/** - 数据处理脚本
- gen_lr.py - 生成低分图像
- extract_and_prepare.py - 数据提取

### 文档目录 (docs/)

**design/** - 技术设计文档
- ARCHITECTURE.md - 系统整体设计
- ALGORITHM.md - 算法和损失函数设计

**tutorial/** - 使用教程
- SETUP.md - 编译和环境配置
- TRAINING.md - 如何训练模型
- INFERENCE.md - 如何进行推理
- GUI.md - GUI 应用使用

**api/** - API 文档
- generator.md - 生成器类接口
- trainer.md - 训练器接口
- inference.md - 推理接口

### 论文相关 (thesis/)

- **figures/** - 论文插图（系统架构、性能对比、结果展示）
- **tables/** - 论文表格（性能指标、消融实验）
- **references/** - 参考文献（BibTeX 格式）

---

## 文件组织原则

### 什么应该上传到 GitHub

✓ 源代码 (include/ + src/)
✓ 配置文件 (config/, CMakeLists.txt)
✓ 文档 (*.md, docs/)
✓ 脚本 (scripts/)
✓ 论文相关 (thesis/)
✓ 许可证和 README

### 什么不应该上传 (在 .gitignore 中)

✗ 编译产物 (build/)
✗ 预训练模型 (checkpoints/*.pt，除了少数关键的)
✗ 数据集 (data/，过大)
✗ IDE 配置 (.vs/, .vscode/, .idea/)
✗ 临时文件 (*.log, *.tmp)

---

## 快速导航

### 对于毕设答辩
1. README.md - 项目概览
2. THESIS_ASSESSMENT.md - 完成度评估
3. FINAL_SUMMARY.md - 项目总结

### 对于想学习的人
1. QUICKSTART.md - 快速开始
2. docs/design/ - 系统设计
3. docs/tutorial/ - 使用教程

### 对于想改进的人
1. ROOT_CAUSE_ANALYSIS.md - 问题分析
2. EVALUATION.md - 评估方法
3. docs/design/ALGORITHM.md - 算法细节

### 对于想复现的人
1. QUICKSTART.md - 环境配置
2. scripts/evaluation/evaluate_model.py - 评估脚本
3. docs/tutorial/TRAINING.md - 训练指南

---

最后更新: 2026-03-23
