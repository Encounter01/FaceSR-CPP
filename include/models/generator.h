#pragma once

// ============================================================================
// generator.h — RRDB生成器网络 (Residual-in-Residual Dense Block Network)
//
// 这是ESRGAN (Enhanced Super-Resolution GAN) 的核心生成器架构。
// 网络结构: 输入(64x64) → 浅层特征 → 23个RRDB块 → 2次上采样 → 输出(256x256)
//
// 与论文的对应关系：
// - 第4章“生成器网络设计”主要讲的就是本文件中的 ResidualDenseBlock、RRDB 和 RRDBNet。
// - 论文最终模型 A4 使用三阶段训练，但推理阶段只需要这里的 RRDBNet 生成器，不需要判别器和损失函数。
// - use_attention 对应 A5/CBAM 消融实验。当前论文结论中 A5 未优于 A4，因此默认结构保持无 attention。
//
// RRDB的设计思想:
// 1. 密集连接(DenseNet): 每层都接收所有前面层的输出，最大化特征复用
// 2. 残差学习(ResNet): 学习残差映射 F(x) = H(x) - x，比直接学H(x)更容易
// 3. 残差缩放: 乘以0.2防止训练不稳定（深网络容易梯度爆炸）
// ============================================================================

#include <torch/torch.h>
#include "models/attention.h"

namespace facesr {

// ============================================================================
// ResidualDenseBlock (RDB) — 残差密集块
//
// 结构: 5层密集连接的卷积 + 残差连接
// 密集连接: 每层的输入 = 原始输入 + 之前所有层的输出（通道拼接）
// 这使得特征通道数逐层递增: 64 → 96 → 128 → 160 → 192
// 最后一层将通道数恢复为64，再通过残差连接与输入相加
// ============================================================================
class ResidualDenseBlockImpl : public torch::nn::Module {
public:
    // 构造函数
    // num_feat: 输入/输出特征通道数（默认64）
    // num_grow_ch: 每层增长的通道数（默认32，DenseNet的growth rate）
    ResidualDenseBlockImpl(int num_feat = 64, int num_grow_ch = 32);

    // 前向传播: 输入[B,C,H,W] → 输出[B,C,H,W]（尺寸不变）
    torch::Tensor forward(torch::Tensor x);

private:
    // 5层3x3卷积，通道数因密集连接逐层递增:
    // conv1: num_feat(64) → num_grow_ch(32)
    // conv2: num_feat + 1*num_grow_ch(96) → num_grow_ch(32)
    // conv3: num_feat + 2*num_grow_ch(128) → num_grow_ch(32)
    // conv4: num_feat + 3*num_grow_ch(160) → num_grow_ch(32)
    // conv5: num_feat + 4*num_grow_ch(192) → num_feat(64) —— 恢复原始通道数
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr},
                      conv4_{nullptr}, conv5_{nullptr};
    double beta_ = 0.2;  // 残差缩放因子，防止深网络训练不稳定
};
// TORCH_MODULE宏: 创建智能指针包装类 ResidualDenseBlock
// 内部是 std::shared_ptr<ResidualDenseBlockImpl>，支持自动内存管理
TORCH_MODULE(ResidualDenseBlock);

// ============================================================================
// RRDB (Residual-in-Residual Dense Block) — 残差中的残差密集块
//
// 由3个RDB串联组成，最后加残差连接。
// "残差中的残差": RRDB本身是残差结构，其内部的RDB也是残差结构。
// 这种嵌套残差设计允许网络非常深。
// 注意：这里的“深”指特征变换路径足够长；实际卷积层数还要结合每个 RDB 的 5 层卷积和主干/上采样卷积理解。
// ============================================================================
class RRDBImpl : public torch::nn::Module {
public:
    RRDBImpl(int num_feat = 64, int num_grow_ch = 32);

    // 前向传播: 输入[B,C,H,W] → 输出[B,C,H,W]（尺寸不变）
    torch::Tensor forward(torch::Tensor x);

private:
    ResidualDenseBlock rdb1_{nullptr}, rdb2_{nullptr}, rdb3_{nullptr};  // 3个子RDB
    double beta_ = 0.2;  // RRDB级别的残差缩放因子
};
TORCH_MODULE(RRDB);

// ============================================================================
// RRDBNet — 完整的RRDB生成器网络
//
// 完整数据流:
// 输入(B,3,64,64)
//   → conv_first: 浅层特征提取 (B,64,64,64)
//   → body: 23个RRDB块 (B,64,64,64) — 深层特征提取
//   → conv_body + 全局残差连接 (B,64,64,64)
//   → conv_up1 + 最近邻2x上采样 (B,64,128,128) — 第1次上采样
//   → conv_up2 + 最近邻2x上采样 (B,64,256,256) — 第2次上采样
//   → conv_hr: HR特征提取 (B,64,256,256)
//   → conv_last: 最终重建 (B,3,256,256)
// ============================================================================
class RRDBNetImpl : public torch::nn::Module {
public:
    // 构造参数:
    // in_channels: 输入通道数（RGB=3）
    // out_channels: 输出通道数（RGB=3）
    // num_feat: 网络特征通道数（默认64，即网络宽度）
    // num_block: RRDB块数量（默认23，即网络深度，ESRGAN标准配置）
    // num_grow_ch: 密集块增长通道数（默认32）
    // scale: 放大倍数（默认4，即64→256）
    // use_attention: 是否在RRDB body后插入CBAM注意力模块；必须和训练保存权重时的结构一致
    RRDBNetImpl(int in_channels = 3, int out_channels = 3,
                int num_feat = 64, int num_block = 23,
                int num_grow_ch = 32, int scale = 4,
                bool use_attention = false);

    // 前向传播
    // 输入: [B, C, H, W] 低分辨率图像
    // 输出: [B, C, H*scale, W*scale] 高分辨率图像
    torch::Tensor forward(torch::Tensor x);

    // 初始化网络权重（Kaiming初始化，适配LeakyReLU激活函数）
    void init_weights();

    // 获取网络总参数量（用于日志输出，典型值约16.7M）
    int64_t get_num_parameters() const;

private:
    int scale_;  // 放大倍数

    // 网络各层:
    torch::nn::Conv2d conv_first_{nullptr};  // 输入层: 3通道RGB → 64特征通道
    torch::nn::Sequential body_;             // 主干: 23个RRDB块的Sequential容器，负责大部分人脸结构和纹理特征恢复
    torch::nn::Conv2d conv_body_{nullptr};   // 主干后卷积: 64 → 64
    torch::nn::Conv2d conv_up1_{nullptr};    // 第1次上采样后卷积: 64 → 64
    torch::nn::Conv2d conv_up2_{nullptr};    // 第2次上采样后卷积: 64 → 64（仅scale=4时）
    torch::nn::Conv2d conv_hr_{nullptr};     // 高分辨率特征提取: 64 → 64
    torch::nn::Conv2d conv_last_{nullptr};   // 最终输出: 64 → 3通道RGB

    // CBAM注意力模块（可选，插入在body后、上采样前）
    CBAM attention_{nullptr};
};
TORCH_MODULE(RRDBNet);

}  // namespace facesr
