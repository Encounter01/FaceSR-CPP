/**
 * @file discriminator.cpp
 * @brief VGG风格判别器网络实现
 *
 * 判别器的阅读重点是空间尺寸变化：
 * 256 -> 128 -> 64 -> 32 -> 16 -> 8，最后展平并输出一个真假分数。
 * 它不负责重建图像，只在 GAN 阶段给生成器提供“更像真实 HR 图像”的训练信号。
 */

#include "models/discriminator.h"
#include <torch/nn/functional.h>

namespace F = torch::nn::functional;

namespace facesr {

VGGStyleDiscriminatorImpl::VGGStyleDiscriminatorImpl(int in_channels, int num_feat, int input_size, bool use_spectral_norm)
    : input_size_(input_size), use_spectral_norm_(use_spectral_norm) {

    // 特征提取层按 VGG 风格堆叠。stride=2 的 4x4 卷积承担下采样，
    // 逐步扩大感受野，让判别器从局部纹理到整体人脸结构都能参与真假判断。
    // 特征提取层
    conv0_0_ = register_module("conv0_0",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_feat, 3).padding(1).bias(true)));
    conv0_1_ = register_module("conv0_1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat, num_feat, 4).stride(2).padding(1).bias(false)));
    bn0_1_ = register_module("bn0_1", torch::nn::BatchNorm2d(num_feat));

    conv1_0_ = register_module("conv1_0",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat, num_feat * 2, 3).padding(1).bias(false)));
    bn1_0_ = register_module("bn1_0", torch::nn::BatchNorm2d(num_feat * 2));
    conv1_1_ = register_module("conv1_1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat * 2, num_feat * 2, 4).stride(2).padding(1).bias(false)));
    bn1_1_ = register_module("bn1_1", torch::nn::BatchNorm2d(num_feat * 2));

    conv2_0_ = register_module("conv2_0",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat * 2, num_feat * 4, 3).padding(1).bias(false)));
    bn2_0_ = register_module("bn2_0", torch::nn::BatchNorm2d(num_feat * 4));
    conv2_1_ = register_module("conv2_1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat * 4, num_feat * 4, 4).stride(2).padding(1).bias(false)));
    bn2_1_ = register_module("bn2_1", torch::nn::BatchNorm2d(num_feat * 4));

    conv3_0_ = register_module("conv3_0",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat * 4, num_feat * 8, 3).padding(1).bias(false)));
    bn3_0_ = register_module("bn3_0", torch::nn::BatchNorm2d(num_feat * 8));
    conv3_1_ = register_module("conv3_1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat * 8, num_feat * 8, 4).stride(2).padding(1).bias(false)));
    bn3_1_ = register_module("bn3_1", torch::nn::BatchNorm2d(num_feat * 8));

    conv4_0_ = register_module("conv4_0",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat * 8, num_feat * 8, 3).padding(1).bias(false)));
    bn4_0_ = register_module("bn4_0", torch::nn::BatchNorm2d(num_feat * 8));
    conv4_1_ = register_module("conv4_1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_feat * 8, num_feat * 8, 4).stride(2).padding(1).bias(false)));
    bn4_1_ = register_module("bn4_1", torch::nn::BatchNorm2d(num_feat * 8));

    // 计算全连接层的输入尺寸。
    // input_size=256 经过 5 次 2 倍下采样后变为 8x8，因此展平特征是 512*8*8。
    // 计算全连接层的输入尺寸
    // input_size=256 经过5次下采样后变为 256/32 = 8
    int fc_input_size = (input_size / 32) * (input_size / 32) * num_feat * 8;

    // 全连接层
    fc1_ = register_module("fc1", torch::nn::Linear(fc_input_size, 100));
    fc2_ = register_module("fc2", torch::nn::Linear(100, 1));

    init_weights();
}

void VGGStyleDiscriminatorImpl::init_weights() {
    for (auto& module : modules(false)) {
        if (auto* conv = module->as<torch::nn::Conv2dImpl>()) {
            torch::nn::init::kaiming_normal_(
                conv->weight, 0.2, torch::kFanIn, torch::kLeakyReLU);
            if (conv->bias.defined()) {
                torch::nn::init::zeros_(conv->bias);
            }
        } else if (auto* bn = module->as<torch::nn::BatchNorm2dImpl>()) {
            torch::nn::init::ones_(bn->weight);
            torch::nn::init::zeros_(bn->bias);
        } else if (auto* linear = module->as<torch::nn::LinearImpl>()) {
            torch::nn::init::kaiming_normal_(
                linear->weight, 0.2, torch::kFanIn, torch::kLeakyReLU);
            torch::nn::init::zeros_(linear->bias);
        }
    }
}

torch::Tensor VGGStyleDiscriminatorImpl::forward(torch::Tensor x) {
    // 检查输入尺寸
    TORCH_CHECK(x.size(2) == input_size_ && x.size(3) == input_size_,
        "Input size should be ", input_size_, "x", input_size_,
        ", but got ", x.size(2), "x", x.size(3));

    // 特征提取
    auto feat = F::leaky_relu(conv0_0_(x), F::LeakyReLUFuncOptions().negative_slope(0.2));
    feat = F::leaky_relu(bn0_1_(conv0_1_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));

    feat = F::leaky_relu(bn1_0_(conv1_0_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    feat = F::leaky_relu(bn1_1_(conv1_1_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));

    feat = F::leaky_relu(bn2_0_(conv2_0_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    feat = F::leaky_relu(bn2_1_(conv2_1_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));

    feat = F::leaky_relu(bn3_0_(conv3_0_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    feat = F::leaky_relu(bn3_1_(conv3_1_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));

    feat = F::leaky_relu(bn4_0_(conv4_0_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));
    feat = F::leaky_relu(bn4_1_(conv4_1_(feat)), F::LeakyReLUFuncOptions().negative_slope(0.2));

    // 展平
    feat = feat.view({feat.size(0), -1});

    // 全连接层
    feat = F::leaky_relu(fc1_(feat), F::LeakyReLUFuncOptions().negative_slope(0.2));
    auto out = fc2_(feat);

    return out;
}

int64_t VGGStyleDiscriminatorImpl::get_num_parameters() const {
    int64_t total = 0;
    for (const auto& param : parameters()) {
        total += param.numel();
    }
    return total;
}

}  // namespace facesr
