# FaceSR_CPP 知识点速查表

> 快速查阅关键概念和技术要点
> 适用于：面试前复习、答辩前准备
>
> 当前代码实现边界见 `../docs/IMPLEMENTATION_STATUS.md`。本速查表已按当前源码修正：不要把 PixelShuffle、转置卷积、Relativistic GAN、完整混合流水线、FP16、梯度裁剪、学习率调度器描述为已实现功能。

---

## 目录

- [深度学习知识点](#深度学习知识点)
- [C++开发知识点](#c开发知识点)
- [计算机视觉知识点](#计算机视觉知识点)
- [软件工程知识点](#软件工程知识点)

---

## 深度学习知识点

### 卷积神经网络（CNN）

**定义**：使用卷积操作提取图像特征的神经网络

**核心概念**：
- 卷积核（Kernel）：3×3、5×5等滑动窗口
- 步长（Stride）：卷积核移动步长
- 填充（Padding）：边缘补零，保持尺寸
- 通道（Channel）：RGB图像有3个通道

**公式**：
```
输出尺寸 = (输入尺寸 - 卷积核尺寸 + 2×填充) / 步长 + 1
```

**本项目应用**：
- RRDB使用3×3卷积
- 64个通道提取特征
- Padding=1保持尺寸

### 残差连接（Residual Connection）

**定义**：跳跃连接，将输入直接加到输出

**公式**：
```
H(x) = F(x) + x
```

**优势**：
- 解决梯度消失
- 支持深层网络（>100层）
- 易于优化

**本项目应用**：
- RRDB有双层残差连接
- 23个RRDB块堆叠
- 梯度可以直接传播

### 密集连接（Dense Connection）

**定义**：每层连接到后面所有层

**公式**：
```
x_l = H([x_0, x_1, ..., x_(l-1)])
```

**优势**：
- 特征复用
- 参数效率高
- 梯度流动好

**本项目应用**：
- 密集块内3层密集连接
- 拼接（Concatenate）而非相加
- 提取多尺度特征

### 生成对抗网络（GAN）

**定义**：生成器和判别器对抗训练

**组成**：
- 生成器（G）：生成假图像
- 判别器（D）：判断真假

**损失函数**：
```
L_D = -log(D(real)) - log(1 - D(fake))
L_G = -log(D(fake))
```

**训练流程**：
1. 固定G，训练D
2. 固定D，训练G
3. 交替进行

**本项目应用**：
- RRDB生成器
- VGG-style判别器
- 默认 Vanilla GAN，可通过配置选择 hinge、lsgan、wgan 等类型

### 感知损失（Perceptual Loss）

**定义**：在特征空间计算损失

**公式**：
```
L_perceptual = ||φ(SR) - φ(HR)||²
φ = VGG19的conv5_4层
```

**优势**：
- 关注语义内容
- 对像素偏移不敏感
- 生成纹理自然

**本项目应用**：
- 使用VGG19提取特征
- conv5_4层（高级特征）
- 权重1.0

### 对抗损失（Adversarial Loss）

**定义**：判别器对生成图像的判断

**公式**：
```
L_adversarial = -log(D(SR))
```

**作用**：
- 生成逼真细节
- 提升感知质量
- 骗过判别器

**本项目应用**：
- 默认权重来自 `config/train_config.ini` 中的 `gan_weight`
- 默认 GAN 类型为 vanilla，`train_config_sharper.ini` 可配置 hinge
- 通过阶段训练逐步引入

### 批归一化（Batch Normalization）

**定义**：对每个batch的特征进行归一化

**公式**：
```
y = γ × (x - μ) / σ + β
```

**优势**：
- 加速训练
- 允许更大学习率
- 减少对初始化的依赖

**注意**：
- 本项目RRDB不使用BN
- 原因：BN会引入伪影
- 替代：使用残差连接稳定训练

### 激活函数

**LeakyReLU**：
```
f(x) = x if x > 0 else α×x
α = 0.2（本项目）
```
- 优势：避免神经元死亡
- 应用：生成器和判别器

**PReLU**：
```
f(x) = x if x > 0 else α×x
α是可学习参数
```
- 优势：自适应调整
- 应用：部分卷积层

**Sigmoid**：
```
f(x) = 1 / (1 + e^(-x))
```
- 应用：判别器输出层

### 上采样（Upsampling）

**方法1：最近邻插值**
- 简单但效果差

**方法2：双线性插值**
- 平滑但模糊

**方法3：最近邻插值 + 卷积（本项目当前实现）**
```
F::interpolate(..., scale_factor=2, mode=nearest)
Conv2d(..., kernel_size=3, padding=1)
```
- 插值负责放大尺寸
- 卷积负责修正和重建特征
- 相比转置卷积更不容易出现棋盘格伪影

**方法4：PixelShuffle（当前代码未使用）**
- 重排像素实现上采样
- 计算效率高

### 梯度裁剪（Gradient Clipping）

**定义**：限制梯度的最大范数

**公式**：
```
if ||g|| > threshold:
    g = g × threshold / ||g||
```

**作用**：
- 防止梯度爆炸
- 稳定训练

**本项目状态**：
- 当前训练代码未实现梯度裁剪
- 可作为后续稳定训练的增强方向

### 学习率调度（Learning Rate Scheduling）

**策略1：阶梯衰减（当前代码未实现）**
```
epoch < 50:  lr = 1e-4
epoch < 150: lr = 5e-5
else:        lr = 1e-5
```

**策略2：余弦退火**
```
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
```

**策略3：指数衰减**
```
lr = lr_0 × γ^epoch
```

### 优化器

**Adam（本项目）**：
```
m_t = β1 × m_(t-1) + (1-β1) × g_t
v_t = β2 × v_(t-1) + (1-β2) × g_t²
θ_t = θ_(t-1) - lr × m_t / (√v_t + ε)
```
- β1 = 0.9（动量）
- β2 = 0.999（二阶动量）
- ε = 1e-8

**优势**：
- 自适应学习率
- 收敛快
- 鲁棒性好

---

## C++开发知识点

### 智能指针

**shared_ptr（共享所有权）**：
```cpp
std::shared_ptr<Generator> gen = std::make_shared<Generator>();
// 引用计数，自动释放
```

**unique_ptr（独占所有权）**：
```cpp
std::unique_ptr<Discriminator> disc = std::make_unique<Discriminator>();
// 不可复制，只能移动
```

**weak_ptr（弱引用）**：
```cpp
std::weak_ptr<Model> weak = shared;
// 不增加引用计数，避免循环引用
```

**本项目应用**：
- 模型使用shared_ptr
- 临时对象使用unique_ptr
- 避免内存泄漏

### 多线程

**std::thread**：
```cpp
std::thread t([]{
    // 线程函数
});
t.join();  // 等待线程结束
```

**std::mutex（互斥锁）**：
```cpp
std::mutex mtx;
std::lock_guard<std::mutex> lock(mtx);
// 自动加锁和解锁
```

**std::condition_variable（条件变量）**：
```cpp
std::condition_variable cv;
cv.wait(lock, []{return ready;});
cv.notify_one();
```

**本项目应用**：
- 数据加载多线程
- 线程池（8线程）
- 生产者-消费者模式

### CMake构建系统

**基本结构**：
```cmake
cmake_minimum_required(VERSION 3.14)
project(FaceSR_CPP)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
```

**常用命令**：
- `find_package`：查找依赖
- `add_executable`：添加可执行文件
- `target_link_libraries`：链接库

**本项目配置**：
- LibTorch路径：CMAKE_PREFIX_PATH
- OpenCV路径：OpenCV_DIR
- C++标准：C++17

### Qt GUI开发

**信号槽机制**：
```cpp
connect(button, &QPushButton::clicked, this, &MainWindow::onButtonClicked);
```

**布局管理**：
```cpp
QVBoxLayout* layout = new QVBoxLayout;
layout->addWidget(button);
setLayout(layout);
```

**文件选择**：
```cpp
QString filePath = QFileDialog::getOpenFileName(
    this, "打开图像", "", "Images (*.png *.jpg *.jpeg *.bmp)");
```

**本项目应用**：
- 文件对话框打开图像
- 实时预览
- 进度条显示

### RAII（资源获取即初始化）

**原则**：
- 构造函数获取资源
- 析构函数释放资源
- 利用栈对象自动管理

**示例**：
```cpp
class FileGuard {
public:
    FileGuard(const std::string& path) {
        file.open(path);
    }
    ~FileGuard() {
        if (file.is_open()) file.close();
    }
private:
    std::ofstream file;
};
```

**本项目应用**：
- 模型加载/卸载
- 文件打开/关闭
- GPU内存管理

### 异常处理

**try-catch**：
```cpp
try {
    model = torch::jit::load(path);
} catch (const c10::Error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return false;
}
```

**自定义异常**：
```cpp
class ModelLoadError : public std::runtime_error {
public:
    ModelLoadError(const std::string& msg) : std::runtime_error(msg) {}
};
```

**本项目应用**：
- 模型加载异常
- 图像读取异常
- 推理异常日志记录
- 当前未实现 OOM 自动重试

### Lambda表达式

**语法**：
```cpp
auto func = [capture](params) -> return_type {
    // 函数体
};
```

**捕获方式**：
- `[=]`：值捕获
- `[&]`：引用捕获
- `[this]`：捕获this指针

**本项目应用**：
```cpp
std::thread t([this]{
    this->process_batch();
});
```

### 移动语义

**右值引用**：
```cpp
void process(std::vector<int>&& data) {
    // data是右值引用
}
```

**std::move**：
```cpp
std::vector<int> a = {1, 2, 3};
std::vector<int> b = std::move(a);  // a被移动，不再有效
```

**优势**：
- 避免不必要的拷贝
- 提升性能

**本项目应用**：
- Tensor移动
- 大对象传递

---

## 计算机视觉知识点

### 图像插值方法

**最近邻插值**：
- 选择最近的像素值
- 速度快但效果差

**双线性插值**：
- 4个邻近像素加权平均
- 平滑但模糊

**双三次插值（Bicubic）**：
- 16个邻近像素加权平均
- 效果好，本项目baseline

**公式**：
```
I(x,y) = Σ Σ a_ij × x^i × y^j
```

### 图像质量评价

**PSNR（峰值信噪比）**：
```
MSE = mean((img1 - img2)²)
PSNR = 10 × log10(255² / MSE)
```
- 单位：dB
- 范围：20-40 dB
- 越大越好

**SSIM（结构相似度）**：
```
SSIM = [l(x,y)]^α × [c(x,y)]^β × [s(x,y)]^γ
l = 亮度比较
c = 对比度比较
s = 结构比较
```
- 范围：0-1
- 越大越好

**LPIPS（感知相似度）**：
```
LPIPS = ||VGG(img1) - VGG(img2)||²
```
- 范围：0-1
- 越小越好

### 边缘检测

**Sobel算子**：
```
Gx = [-1 0 1]    Gy = [-1 -2 -1]
     [-2 0 2]         [ 0  0  0]
     [-1 0 1]         [ 1  2  1]

G = √(Gx² + Gy²)
```

**本项目应用**：
- 计算边缘密度
- 分层评估依据

### 颜色空间

**RGB**：
- 红绿蓝三通道
- 范围：0-255

**HSV**：
- 色调、饱和度、明度
- 更符合人眼感知

**YCbCr**：
- 亮度、色度
- 视频压缩常用

**本项目应用**：
- 使用RGB空间
- 计算亮度：0.299×R + 0.587×G + 0.114×B

### 图像增强

**直方图均衡化**：
- 增强对比度
- 拉伸灰度范围

**锐化**：
- 增强边缘
- 提升清晰度

**去噪**：
- 高斯滤波
- 双边滤波

### 特征提取

**VGG特征**：
- 使用VGG19网络
- 提取conv5_4层
- 用于感知损失

**HOG特征**：
- 方向梯度直方图
- 用于目标检测

**SIFT特征**：
- 尺度不变特征
- 用于图像匹配

---

## 软件工程知识点

### 配置管理

**INI文件格式**：
```ini
[section]
key = value
```

**优势**：
- 易读易写
- 层次清晰
- 版本管理方便

**本项目应用**：
- 模型配置
- 训练配置
- 数据配置

### 日志系统设计

**日志级别**：
- DEBUG：调试信息
- INFO：一般信息
- WARNING：警告
- ERROR：错误

**日志格式**：
```
[LEVEL] [TIMESTAMP] MESSAGE
```

**本项目应用**：
- 训练进度记录
- 错误追踪
- 训练和推理状态记录

### 模型版本管理

**命名规范**：
```
generator_epochN.pt
discriminator_epochN.pt
generator_latest.pt
generator_best.pt
train_state.bin
optimizer_g_state.pt
optimizer_d_state.pt
```

**元信息**：
- epoch数
- best_psnr
- global_step
- 优化器状态
- 当前代码没有生成 JSON 元信息文件

**策略**：
- latest：最新模型
- best：最佳模型
- 定期保存

### 代码模块化设计

**原则**：
- 单一职责
- 高内聚低耦合
- 接口抽象

**本项目模块**：
- models：模型定义
- utils：工具函数
- trainer：训练逻辑
- inference：推理接口

### 设计模式

**单例模式**：
```cpp
class Singleton {
    static Singleton* instance;
    Singleton() {}
public:
    static Singleton* getInstance();
};
```

**工厂模式**：
```cpp
class ModelFactory {
public:
    static Model* create(const std::string& type);
};
```

**策略模式**：
```cpp
class ILoss {
public:
    virtual float compute() = 0;
};
```

### 性能优化

**原则**：
1. 先测量后优化
2. 优化热点代码
3. 避免过早优化

**方法**：
- Profiling分析
- 减少内存拷贝
- 并行计算
- 缓存优化

**本项目优化**：
- C++实现
- GPU加速
- 批量处理
- 内存池

### 测试策略

**单元测试**：
- 测试单个函数
- 使用Google Test

**集成测试**：
- 测试模块交互
- 端到端测试

**性能测试**：
- 测试推理速度
- 测试内存占用

### 文档编写

**类型**：
- README：项目概述
- QUICKSTART：快速开始
- API文档：接口说明
- 设计文档：架构设计

**原则**：
- 清晰简洁
- 示例丰富
- 及时更新

**本项目文档**：
- 9篇技术文档
- 3篇评估报告
- 完整教程手册

---

## 快速记忆口诀

**RRDB结构**：
```
残差套残差，密集在中间
三层嵌套结构，梯度畅通无阻
```

**三阶段训练**：
```
先学内容后学细节
逐步引入对抗损失
学习率同步衰减
```

**三大损失**：
```
L1保内容，感知保纹理
对抗生细节，权重要调好
```

**三大指标**：
```
PSNR看像素，SSIM看结构
LPIPS看感知，各有侧重点
```

**性能优化**：
```
C++快一倍，GPU快十倍
批量可扩展，流水线待完善
```

---

**知识点速查表完成！📚**
