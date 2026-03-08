/**
 * @file generator.cpp
 * @brief RRDB生成器网络实现
 */

#include "models/generator.h"
#include <torch/nn/functional.h>

namespace F = torch::nn::functional;

namespace facesr {
namespace models {

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

    // 初始化权重
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
    auto x1 = F::leaky_relu(conv1_(x), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto x2 = F::leaky_relu(conv2_(torch::cat({x, x1}, 1)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto x3 = F::leaky_relu(conv3_(torch::cat({x, x1, x2}, 1)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto x4 = F::leaky_relu(conv4_(torch::cat({x, x1, x2, x3}, 1)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto x5 = conv5_(torch::cat({x, x1, x2, x3, x4}, 1));

    // 残差连接
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
                         int num_grow_ch, int scale)
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
    // 浅层特征
    auto feat = conv_first_(x);

    // RRDB主干
    auto body_feat = conv_body_(body_->forward(feat));

    // 全局残差连接
    feat = feat + body_feat;

    // 上采样 (使用最近邻插值 + 卷积)
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

}  // namespace models
}  // namespace facesr
