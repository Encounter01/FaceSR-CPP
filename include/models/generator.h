#pragma once
/**
 * @file generator.h
 * @brief RRDB生成器网络 (Residual-in-Residual Dense Block)
 *
 * 基于ESRGAN的生成器架构，用于人脸图像超分辨率重建
 */

#include <torch/torch.h>
#include <vector>

namespace facesr {
namespace models {

/**
 * @brief 残差密集块 (Residual Dense Block)
 *
 * 使用密集连接提取丰富的特征
 */
class ResidualDenseBlockImpl : public torch::nn::Module {
public:
    /**
     * @param num_feat 输入特征通道数
     * @param num_grow_ch 每层增长的通道数
     */
    ResidualDenseBlockImpl(int num_feat = 64, int num_grow_ch = 32);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr},
                      conv4_{nullptr}, conv5_{nullptr};
    double beta_ = 0.2;  // 残差缩放因子
};
TORCH_MODULE(ResidualDenseBlock);


/**
 * @brief Residual-in-Residual Dense Block (RRDB)
 *
 * 由3个残差密集块组成的残差块
 */
class RRDBImpl : public torch::nn::Module {
public:
    /**
     * @param num_feat 特征通道数
     * @param num_grow_ch 增长通道数
     */
    RRDBImpl(int num_feat = 64, int num_grow_ch = 32);

    torch::Tensor forward(torch::Tensor x);

private:
    ResidualDenseBlock rdb1_{nullptr}, rdb2_{nullptr}, rdb3_{nullptr};
    double beta_ = 0.2;
};
TORCH_MODULE(RRDB);


/**
 * @brief RRDB网络 - 用于图像超分辨率的生成器
 *
 * 网络结构:
 * 输入 -> 浅层特征提取 -> N个RRDB块 -> 上采样 -> 输出
 */
class RRDBNetImpl : public torch::nn::Module {
public:
    /**
     * @param in_channels 输入图像通道数
     * @param out_channels 输出图像通道数
     * @param num_feat 中间特征通道数
     * @param num_block RRDB块的数量
     * @param num_grow_ch 密集块中每层增长的通道数
     * @param scale 上采样倍数 (2 或 4)
     */
    RRDBNetImpl(int in_channels = 3, int out_channels = 3,
                int num_feat = 64, int num_block = 23,
                int num_grow_ch = 32, int scale = 4);

    /**
     * @brief 前向传播
     * @param x 输入低分辨率图像 [B, C, H, W]
     * @return 输出高分辨率图像 [B, C, H*scale, W*scale]
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief 获取参数数量
     */
    int64_t get_num_parameters() const;

private:
    void init_weights();

    int scale_;
    torch::nn::Conv2d conv_first_{nullptr};
    torch::nn::Sequential body_{nullptr};
    torch::nn::Conv2d conv_body_{nullptr};
    torch::nn::Conv2d conv_up1_{nullptr};
    torch::nn::Conv2d conv_up2_{nullptr};
    torch::nn::Conv2d conv_hr_{nullptr};
    torch::nn::Conv2d conv_last_{nullptr};
};
TORCH_MODULE(RRDBNet);

}  // namespace models
}  // namespace facesr
