#pragma once
/**
 * @file discriminator.h
 * @brief VGG风格判别器网络
 *
 * 用于GAN对抗训练的判别器
 */

#include <torch/torch.h>

namespace facesr {
namespace models {

/**
 * @brief VGG风格判别器
 *
 * 使用多层卷积逐步提取特征，最终输出真假概率
 */
class VGGStyleDiscriminatorImpl : public torch::nn::Module {
public:
    /**
     * @param in_channels 输入图像通道数
     * @param num_feat 基础特征通道数
     * @param input_size 输入图像尺寸
     */
    VGGStyleDiscriminatorImpl(int in_channels = 3, int num_feat = 64, int input_size = 256);

    /**
     * @brief 前向传播
     * @param x 输入图像 [B, C, H, W]
     * @return 判别结果 [B, 1]
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief 获取参数数量
     */
    int64_t get_num_parameters() const;

private:
    void init_weights();

    int input_size_;

    // 特征提取层
    torch::nn::Conv2d conv0_0_{nullptr}, conv0_1_{nullptr};
    torch::nn::BatchNorm2d bn0_1_{nullptr};

    torch::nn::Conv2d conv1_0_{nullptr}, conv1_1_{nullptr};
    torch::nn::BatchNorm2d bn1_0_{nullptr}, bn1_1_{nullptr};

    torch::nn::Conv2d conv2_0_{nullptr}, conv2_1_{nullptr};
    torch::nn::BatchNorm2d bn2_0_{nullptr}, bn2_1_{nullptr};

    torch::nn::Conv2d conv3_0_{nullptr}, conv3_1_{nullptr};
    torch::nn::BatchNorm2d bn3_0_{nullptr}, bn3_1_{nullptr};

    torch::nn::Conv2d conv4_0_{nullptr}, conv4_1_{nullptr};
    torch::nn::BatchNorm2d bn4_0_{nullptr}, bn4_1_{nullptr};

    // 全连接层
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};
TORCH_MODULE(VGGStyleDiscriminator);

}  // namespace models
}  // namespace facesr
