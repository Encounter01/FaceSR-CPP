#pragma once

// ============================================================================
// discriminator.h — VGG风格判别器网络
//
// 作用: 判断输入的256x256图像是真实的高分辨率图像还是生成器生成的假图像
//
// 阅读提示：
// - 判别器只在训练第三阶段参与，用来提供 GAN 对抗损失。
// - 推理阶段不会加载判别器；最终应用只需要生成器。
// - 本类输出 logits，不在 forward 末尾手动 sigmoid，具体损失函数会选择 BCEWithLogits、LSGAN、Hinge 等形式。
//
// 网络结构借鉴VGG网络设计:
// - 5组卷积块，每组2个3x3卷积(一个stride=1，一个stride=2用于下采样)
// - 通道数按 64→128→256→512→512 递增
// - 空间尺寸按 256→128→64→32→16→8 递减
// - 最后通过全连接层输出单个标量（真/假判别分数）
//
// 与VGG分类网络的区别:
// - 使用LeakyReLU而非ReLU（允许负值有小梯度，训练更稳定）
// - 输出1个值而非1000个类别
// - 使用BatchNorm加速训练（第0组除外，因为输入层不需要BN）
// ============================================================================

#include <torch/torch.h>

namespace facesr {

class VGGStyleDiscriminatorImpl : public torch::nn::Module {
public:
    // 构造参数:
    // in_channels: 输入通道数（默认3, RGB图像）
    // num_feat: 第一组的特征通道数（默认64，后续组逐渐翻倍）
    // input_size: 输入图像尺寸（默认256x256，影响全连接层的输入维度）
    // use_spectral_norm: 谱归一化预留开关；当前实现保存该标志，但卷积层仍按普通 Conv2d + BatchNorm 构造
    VGGStyleDiscriminatorImpl(int in_channels = 3, int num_feat = 64,
                               int input_size = 256, bool use_spectral_norm = false);

    // 初始化权重
    void init_weights();

    // 前向传播
    // 输入: [B, 3, 256, 256] 图像
    // 输出: [B, 1] 判别分数（未经sigmoid，logits形式）
    torch::Tensor forward(torch::Tensor x);

    // 获取参数量（典型值约8.3M）
    int64_t get_num_parameters() const;

private:
    int input_size_;          // 输入尺寸，用于计算FC层维度
    bool use_spectral_norm_;  // 是否使用谱归一化

    // 5组特征提取层，每组包含2个卷积+BatchNorm:
    // 第0组: 3→64, stride=1 (256x256) + 64→64, stride=2 (→128x128)
    // 注意: conv0_0不使用BN（输入层直接做BN通常没有意义）
    torch::nn::Conv2d conv0_0_{nullptr}, conv0_1_{nullptr};
    torch::nn::BatchNorm2d bn0_1_{nullptr};

    // 第1组: 64→128, stride=1 (128x128) + 128→128, stride=2 (→64x64)
    torch::nn::Conv2d conv1_0_{nullptr}, conv1_1_{nullptr};
    torch::nn::BatchNorm2d bn1_0_{nullptr}, bn1_1_{nullptr};

    // 第2组: 128→256, stride=1 (64x64) + 256→256, stride=2 (→32x32)
    torch::nn::Conv2d conv2_0_{nullptr}, conv2_1_{nullptr};
    torch::nn::BatchNorm2d bn2_0_{nullptr}, bn2_1_{nullptr};

    // 第3组: 256→512, stride=1 (32x32) + 512→512, stride=2 (→16x16)
    torch::nn::Conv2d conv3_0_{nullptr}, conv3_1_{nullptr};
    torch::nn::BatchNorm2d bn3_0_{nullptr}, bn3_1_{nullptr};

    // 第4组: 512→512, stride=1 (16x16) + 512→512, stride=2 (→8x8)
    torch::nn::Conv2d conv4_0_{nullptr}, conv4_1_{nullptr};
    torch::nn::BatchNorm2d bn4_0_{nullptr}, bn4_1_{nullptr};

    // 全连接层: 展平后 512×8×8=32768 → 100 → 1
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};
TORCH_MODULE(VGGStyleDiscriminator);

}  // namespace facesr
