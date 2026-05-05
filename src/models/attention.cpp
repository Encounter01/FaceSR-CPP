#include "models/attention.h"
#include <torch/nn/functional.h>

namespace F = torch::nn::functional;

namespace facesr {

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
    x = x * channel_att_(x);
    x = x * spatial_att_(x);
    return x;
}

}  // namespace facesr
