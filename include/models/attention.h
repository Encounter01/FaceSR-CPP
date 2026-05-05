#pragma once
#include <torch/torch.h>

namespace facesr {

class ChannelAttentionImpl : public torch::nn::Module {
public:
    explicit ChannelAttentionImpl(int num_channels, int reduction = 16);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
    int num_channels_;
};
TORCH_MODULE(ChannelAttention);

class SpatialAttentionImpl : public torch::nn::Module {
public:
    explicit SpatialAttentionImpl(int kernel_size = 7);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Conv2d conv_{nullptr};
};
TORCH_MODULE(SpatialAttention);

class CBAMImpl : public torch::nn::Module {
public:
    explicit CBAMImpl(int num_channels, int reduction = 16, int kernel_size = 7);
    torch::Tensor forward(torch::Tensor x);
private:
    ChannelAttention channel_att_{nullptr};
    SpatialAttention spatial_att_{nullptr};
};
TORCH_MODULE(CBAM);

}  // namespace facesr
