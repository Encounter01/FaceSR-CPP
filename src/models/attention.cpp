#include "models/attention.h"
#include <torch/nn/functional.h>

namespace F = torch::nn::functional;

namespace facesr {

// 通道注意力关注“哪些特征通道更重要”。
// 平均池化提供整体响应，最大池化保留强响应，两路结果共享 MLP 后相加得到通道权重。
ChannelAttentionImpl::ChannelAttentionImpl(int num_channels, int reduction)
    : num_channels_(num_channels) {
    int mid = std::max(1, num_channels / reduction);
    fc1_ = register_module("fc1", torch::nn::Linear(
        torch::nn::LinearOptions(num_channels, mid).bias(true)));
    fc2_ = register_module("fc2", torch::nn::Linear(
        torch::nn::LinearOptions(mid, num_channels).bias(true)));
}

torch::Tensor ChannelAttentionImpl::forward(torch::Tensor x) {
    auto avg = torch::adaptive_avg_pool2d(x, {1, 1}).view({x.size(0), -1});
    auto mx  = std::get<0>(torch::adaptive_max_pool2d(x, {1, 1})).view({x.size(0), -1});
    auto scale = torch::sigmoid(fc2_(F::relu(fc1_(avg))) + fc2_(F::relu(fc1_(mx))));
    return scale.view({x.size(0), num_channels_, 1, 1});
}

// 空间注意力关注“图像空间中哪些位置更重要”。
// 沿通道做平均/最大聚合后拼接，再通过一个卷积得到 HxW 的空间权重图。
SpatialAttentionImpl::SpatialAttentionImpl(int kernel_size) {
    conv_ = register_module("conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(2, 1, kernel_size)
            .padding((kernel_size - 1) / 2)
            .bias(false)));
}

torch::Tensor SpatialAttentionImpl::forward(torch::Tensor x) {
    auto avg = x.mean(1, /*keepdim=*/true);
    auto mx  = std::get<0>(torch::max(x, 1, /*keepdim=*/true));
    return torch::sigmoid(conv_(torch::cat({avg, mx}, 1)));
}

CBAMImpl::CBAMImpl(int num_channels, int reduction, int kernel_size) {
    channel_att_ = register_module("channel_att", ChannelAttention(num_channels, reduction));
    spatial_att_ = register_module("spatial_att", SpatialAttention(kernel_size));
}

torch::Tensor CBAMImpl::forward(torch::Tensor x) {
    // CBAM 的标准顺序是 channel -> spatial。这里直接乘回原特征，
    // 因此输入输出尺寸不变，可以无缝插在 RRDB 主干和上采样模块之间。
    x = x * channel_att_(x);
    x = x * spatial_att_(x);
    return x;
}

}  // namespace facesr
