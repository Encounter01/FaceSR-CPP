#pragma once
/**
 * @file losses.h
 * @brief 损失函数模块
 *
 * 包含用于人脸超分辨率重建的各种损失函数
 */

#include <torch/torch.h>
#include <string>
#include <map>
#include "common/config.h"

namespace facesr {
namespace models {

/**
 * @brief 像素级损失 (L1/L2 Loss)
 */
class PixelLoss : public torch::nn::Module {
public:
    /**
     * @param loss_type 损失类型
     */
    explicit PixelLoss(PixelLossType loss_type = PixelLossType::L1);

    /**
     * @brief 兼容旧API的构造函数
     * @deprecated 推荐使用枚举类型
     */
    explicit PixelLoss(const std::string& loss_type);

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

    /**
     * @brief 获取损失类型
     */
    PixelLossType getLossType() const { return loss_type_; }

private:
    PixelLossType loss_type_;
};


/**
 * @brief VGG特征提取器 (用于感知损失)
 */
class VGGFeatureExtractor : public torch::nn::Module {
public:
    /**
     * @param feature_layer 使用VGG的哪一层特征
     * @param use_input_norm 是否对输入进行标准化
     */
    explicit VGGFeatureExtractor(int feature_layer = constants::DEFAULT_VGG_FEATURE_LAYER,
                                  bool use_input_norm = true);

    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief 加载预训练VGG权重
     * @param weight_path 权重文件路径
     */
    void load_pretrained(const std::string& weight_path);

private:
    void buildVGG19(int feature_layer);
    void initImageNetNorm();

    bool use_input_norm_;
    int feature_layer_;
    torch::nn::Sequential features_{nullptr};
    torch::Tensor mean_;
    torch::Tensor std_;
};


/**
 * @brief 感知损失 (Perceptual Loss)
 */
class PerceptualLoss : public torch::nn::Module {
public:
    /**
     * @param feature_layer VGG特征层
     * @param loss_type 损失类型
     */
    explicit PerceptualLoss(int feature_layer = constants::DEFAULT_VGG_FEATURE_LAYER,
                            PixelLossType loss_type = PixelLossType::L1);

    /**
     * @brief 兼容旧API的构造函数
     */
    PerceptualLoss(int feature_layer, const std::string& loss_type);

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

    void load_vgg_weights(const std::string& weight_path);

private:
    PixelLossType loss_type_;
    std::shared_ptr<VGGFeatureExtractor> vgg_;
};


/**
 * @brief GAN对抗损失
 */
class GANLoss : public torch::nn::Module {
public:
    /**
     * @param gan_type GAN类型
     * @param real_label_val 真实标签值
     * @param fake_label_val 假标签值
     */
    explicit GANLoss(GANType gan_type = GANType::Vanilla,
                     double real_label_val = constants::REAL_LABEL_VALUE,
                     double fake_label_val = constants::FAKE_LABEL_VALUE);

    /**
     * @brief 兼容旧API的构造函数
     */
    GANLoss(const std::string& gan_type,
            double real_label_val = constants::REAL_LABEL_VALUE,
            double fake_label_val = constants::FAKE_LABEL_VALUE);

    /**
     * @brief 计算GAN损失
     * @param input 判别器输出
     * @param target_is_real 是否为真实样本
     * @param is_disc 是否是判别器的损失
     */
    torch::Tensor forward(torch::Tensor input, bool target_is_real, bool is_disc = false);

    /**
     * @brief 获取GAN类型
     */
    GANType getGANType() const { return gan_type_; }

private:
    torch::Tensor getTargetLabel(torch::Tensor input, bool target_is_real);
    torch::Tensor computeVanillaLoss(torch::Tensor input, torch::Tensor target);
    torch::Tensor computeLSGANLoss(torch::Tensor input, torch::Tensor target);
    torch::Tensor computeWGANLoss(torch::Tensor input, bool target_is_real);
    torch::Tensor computeHingeLoss(torch::Tensor input, bool target_is_real, bool is_disc);

    GANType gan_type_;
    double real_label_val_;
    double fake_label_val_;
};


/**
 * @brief 组合损失
 */
class CombinedLoss : public torch::nn::Module {
public:
    /**
     * @param pixel_weight 像素损失权重
     * @param perceptual_weight 感知损失权重
     * @param gan_weight GAN损失权重
     * @param pixel_loss_type 像素损失类型
     * @param gan_type GAN类型
     */
    CombinedLoss(double pixel_weight = constants::DEFAULT_PIXEL_WEIGHT,
                 double perceptual_weight = constants::DEFAULT_PERCEPTUAL_WEIGHT,
                 double gan_weight = constants::DEFAULT_GAN_WEIGHT,
                 PixelLossType pixel_loss_type = PixelLossType::L1,
                 GANType gan_type = GANType::Vanilla);

    /**
     * @brief 计算组合损失
     * @param pred 预测图像
     * @param target 目标图像
     * @param disc_pred_fake 判别器对假图像的预测（可选）
     * @param use_perceptual 是否使用感知损失
     * @param use_gan 是否使用GAN损失
     * @return 包含各项损失的map
     */
    std::map<std::string, torch::Tensor> forward(
        torch::Tensor pred,
        torch::Tensor target,
        torch::Tensor disc_pred_fake = {},
        bool use_perceptual = true,
        bool use_gan = false);

    void load_vgg_weights(const std::string& weight_path);

    // Getter方法
    double getPixelWeight() const { return pixel_weight_; }
    double getPerceptualWeight() const { return perceptual_weight_; }
    double getGANWeight() const { return gan_weight_; }

    // Setter方法
    void setPixelWeight(double weight) { pixel_weight_ = weight; }
    void setPerceptualWeight(double weight) { perceptual_weight_ = weight; }
    void setGANWeight(double weight) { gan_weight_ = weight; }

private:
    double pixel_weight_;
    double perceptual_weight_;
    double gan_weight_;

    std::shared_ptr<PixelLoss> pixel_loss_;
    std::shared_ptr<PerceptualLoss> perceptual_loss_;
    std::shared_ptr<GANLoss> gan_loss_;
};

}  // namespace models
}  // namespace facesr
