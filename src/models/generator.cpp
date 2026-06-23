/**
 * @file generator.cpp
 * @brief RRDB生成器网络实现
 *
 * 阅读顺序：
 * 1. ResidualDenseBlockImpl::forward()：看密集连接如何把 x、x1、x2... 按通道拼接。
 * 2. RRDBImpl::forward()：看 3 个 RDB 如何叠加成“残差中的残差”。
 * 3. RRDBNetImpl::forward()：看完整 4 倍超分流程如何从 LR 特征到 SR 图像。
 */

#include "models/generator.h"
#include <torch/nn/functional.h>

namespace F = torch::nn::functional;

namespace facesr {

// ==================== ResidualDenseBlock ====================

ResidualDenseBlockImpl::ResidualDenseBlockImpl(int num_feat, int num_grow_ch) {
    conv1_ = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat, num_grow_ch, 3).padding(1)));
    conv2_ = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat + num_grow_ch, num_grow_ch, 3).padding(1)));
    conv3_ = register_module("conv3",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat + 2 * num_grow_ch, num_grow_ch, 3).padding(1)));
    conv4_ = register_module("conv4",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat + 3 * num_grow_ch, num_grow_ch, 3).padding(1)));
    conv5_ = register_module("conv5",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat + 4 * num_grow_ch, num_feat, 3).padding(1)));

    // 初始化权重。
    // Kaiming 初始化适配 LeakyReLU；最后一层零初始化让 RDB 初始时近似恒等映射，
    // 这样深层残差块刚开始训练时不会过早扰动主干特征。
    for (auto& module : modules(false)) {
        if (auto* conv = module->as<torch::nn::Conv2dImpl>()) {
            torch::nn::init::kaiming_normal_(
                conv->weight, 0.2, torch::kFanIn, torch::kLeakyReLU);
            if (conv->bias.defined()) {
                torch::nn::init::zeros_(conv->bias);
            }
        }
    }
    // 最后一层使用零初始化
    torch::nn::init::zeros_(conv5_->weight);
}

torch::Tensor ResidualDenseBlockImpl::forward(torch::Tensor x) {
    // RDB 的关键是“密集连接”：后续卷积层不只接收上一层输出，
    // 还接收原始输入和所有前面层的输出。这样浅层边缘、局部纹理和深层语义特征能被反复复用。
    auto x1 = F::leaky_relu(conv1_(x), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto x2 = F::leaky_relu(conv2_(torch::cat({x, x1}, 1)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto x3 = F::leaky_relu(conv3_(torch::cat({x, x1, x2}, 1)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto x4 = F::leaky_relu(conv4_(torch::cat({x, x1, x2, x3}, 1)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto x5 = conv5_(torch::cat({x, x1, x2, x3, x4}, 1));

    // 残差缩放后再加回输入。0.2 是 ESRGAN/RRDB 中常用的稳定训练技巧，
    // 论文中也用它解释深层网络避免数值震荡的原因。
    return x5 * beta_ + x;
}


// ==================== RRDB ====================

RRDBImpl::RRDBImpl(int num_feat, int num_grow_ch) {
    rdb1_ = register_module("rdb1", ResidualDenseBlock(num_feat, num_grow_ch));
    rdb2_ = register_module("rdb2", ResidualDenseBlock(num_feat, num_grow_ch));
    rdb3_ = register_module("rdb3", ResidualDenseBlock(num_feat, num_grow_ch));
}

torch::Tensor RRDBImpl::forward(torch::Tensor x) {
    auto out = rdb1_(x);
    out = rdb2_(out);
    out = rdb3_(out);
    return out * beta_ + x;
}


// ==================== RRDBNet ====================

RRDBNetImpl::RRDBNetImpl(int in_channels, int out_channels,
                         int num_feat, int num_block,
                         int num_grow_ch, int scale,
                         bool use_attention)
    : scale_(scale) {

    // 浅层特征提取
    conv_first_ = register_module("conv_first",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_feat, 3).padding(1)));

    // RRDB主干网络
    body_ = register_module("body", torch::nn::Sequential());
    for (int i = 0; i < num_block; ++i) {
        body_->push_back(RRDB(num_feat, num_grow_ch));
    }

    // 主干后的卷积
    conv_body_ = register_module("conv_body",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat, num_feat, 3).padding(1)));

    // 上采样模块
    conv_up1_ = register_module("conv_up1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat, num_feat, 3).padding(1)));

    if (scale_ == 4) {
        conv_up2_ = register_module("conv_up2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat, num_feat, 3).padding(1)));
    }

    // 高分辨率特征提取和重建
    conv_hr_ = register_module("conv_hr",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat, num_feat, 3).padding(1)));
    conv_last_ = register_module("conv_last",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat, out_channels, 3).padding(1)));

    if (use_attention) {
        attention_ = register_module("attention", CBAM(num_feat));
    }

    init_weights();
}

void RRDBNetImpl::init_weights() {
    for (auto& module : {conv_first_, conv_body_, conv_up1_, conv_hr_, conv_last_}) {
        torch::nn::init::kaiming_normal_(
            module->weight, 0.2, torch::kFanIn, torch::kLeakyReLU);
        if (module->bias.defined()) {
            torch::nn::init::zeros_(module->bias);
        }
    }

    if (scale_ == 4 && conv_up2_) {
        torch::nn::init::kaiming_normal_(
            conv_up2_->weight, 0.2, torch::kFanIn, torch::kLeakyReLU);
        if (conv_up2_->bias.defined()) {
            torch::nn::init::zeros_(conv_up2_->bias);
        }
    }
}

torch::Tensor RRDBNetImpl::forward(torch::Tensor x) {
    // 浅层特征：把 RGB 图像映射到 64 通道特征空间，后续 RRDB 都在该特征空间内工作。
    auto feat = conv_first_(x);

    // RRDB 主干：论文中的核心特征提取部分。
    // conv_body 后与浅层特征做全局残差，保证网络主要学习 LR 到 HR 之间缺失的高频增量。
    auto body_feat = conv_body_(body_->forward(feat));

    // 全局残差连接
    feat = feat + body_feat;

    if (attention_) {
        // CBAM 只在启用 attention 的模型中生效；推理时该开关必须和训练 checkpoint 匹配。
        feat = attention_(feat);
    }

    // 上采样使用“最近邻插值 + 卷积”，不是反卷积。
    // 这样实现简单稳定，也避免转置卷积在超分辨率中常见的棋盘格伪影。
    feat = F::interpolate(feat, F::InterpolateFuncOptions()
        .scale_factor(std::vector<double>{2.0, 2.0})
        .mode(torch::kNearest));
    feat = F::leaky_relu(conv_up1_(feat), F::LeakyReLUFuncOptions().negative_slope(0.2));

    if (scale_ == 4) {
        feat = F::interpolate(feat, F::InterpolateFuncOptions()
            .scale_factor(std::vector<double>{2.0, 2.0})
            .mode(torch::kNearest));
        feat = F::leaky_relu(conv_up2_(feat), F::LeakyReLUFuncOptions().negative_slope(0.2));
    }

    // 高分辨率特征提取和重建
    auto out = F::leaky_relu(conv_hr_(feat), F::LeakyReLUFuncOptions().negative_slope(0.2));
    out = conv_last_(out);

    return out;
}

int64_t RRDBNetImpl::get_num_parameters() const {
    int64_t total = 0;
    for (const auto& param : parameters()) {
        total += param.numel();
    }
    return total;
}

}  // namespace facesr
