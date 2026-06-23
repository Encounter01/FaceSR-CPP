#pragma once
#include <torch/torch.h>

namespace facesr {

// CBAM (Convolutional Block Attention Module) 是可选增强模块。
// 在本项目中它对应论文 A5 消融实验：先做通道注意力，再做空间注意力。
// 注意：如果训练时启用了 CBAM，推理时也必须用相同结构加载权重；论文最终采用的 A4 模型默认不依赖它。

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
