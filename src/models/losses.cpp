/**
 * @file losses.cpp
 * @brief 损失函数模块实现
 *
 * 读这个文件时建议按训练阶段理解：
 * 1. PixelLoss 是所有阶段都启用的基础重建约束。
 * 2. PerceptualLoss 在第二阶段开始启用，比较 VGG 特征而不是直接比较像素。
 * 3. GANLoss 在第三阶段启用，使用判别器输出参与生成器和判别器的对抗目标。
 */

#include "models/losses.h"
#include "common/logger.h"
#include <torch/nn/functional.h>
#include <torch/script.h>

namespace F = torch::nn::functional;

namespace facesr {
namespace models {

// ==================== PixelLoss ====================

PixelLoss::PixelLoss(PixelLossType loss_type)
    : loss_type_(loss_type) {
}

PixelLoss::PixelLoss(const std::string& loss_type)
    : loss_type_(stringToPixelLossType(loss_type)) {
}

torch::Tensor PixelLoss::forward(torch::Tensor pred, torch::Tensor target) {
    // L1 对异常像素不如 L2 敏感，超分辨率中常用于保持整体结构并减少过度平滑。
    switch (loss_type_) {
        case PixelLossType::L1:
            return F::l1_loss(pred, target);
        case PixelLossType::L2:
            return F::mse_loss(pred, target);
        default:
            TORCH_CHECK(false, "Unsupported loss type");
            return {};
    }
}


// ==================== VGGFeatureExtractor ====================

VGGFeatureExtractor::VGGFeatureExtractor(int feature_layer, bool use_input_norm)
    : use_input_norm_(use_input_norm), feature_layer_(feature_layer) {

    buildVGG19(feature_layer);
    initImageNetNorm();

    // 冻结参数
    for (auto& param : parameters()) {
        param.set_requires_grad(false);
    }

    LOG_DEBUG("VGGFeatureExtractor created with feature_layer=", feature_layer);
}

void VGGFeatureExtractor::buildVGG19(int feature_layer) {
    features_ = register_module("features", torch::nn::Sequential());

    // VGG19 结构: [64, 64, M, 128, 128, M, 256, 256, 256, 256, M, 512, 512, 512, 512, M, 512, 512, 512, 512]
    // M 表示 MaxPool
    std::vector<int> channels = {64, 64, 128, 128, 256, 256, 256, 256,
                                  512, 512, 512, 512, 512, 512, 512, 512};
    int in_channels = 3;
    int num_layers = std::min(static_cast<int>(channels.size()), feature_layer / 2);

    for (int i = 0; i < num_layers; ++i) {
        features_->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, channels[i], 3).padding(1)));
        features_->push_back(torch::nn::Functional(torch::relu));

        // 每2层后添加池化
        if ((i + 1) % 2 == 0 && i < static_cast<int>(channels.size()) - 1) {
            features_->push_back(torch::nn::Functional([](torch::Tensor x) {
                return F::max_pool2d(x, F::MaxPool2dFuncOptions(2).stride(2));
            }));
        }

        in_channels = channels[i];
    }
}

void VGGFeatureExtractor::initImageNetNorm() {
    mean_ = register_buffer("mean",
        torch::tensor({constants::IMAGENET_MEAN_R,
                       constants::IMAGENET_MEAN_G,
                       constants::IMAGENET_MEAN_B}).view({1, 3, 1, 1}));
    std_ = register_buffer("std",
        torch::tensor({constants::IMAGENET_STD_R,
                       constants::IMAGENET_STD_G,
                       constants::IMAGENET_STD_B}).view({1, 3, 1, 1}));
}

torch::Tensor VGGFeatureExtractor::forward(torch::Tensor x) {
    if (use_input_norm_) {
        // 预训练 VGG 通常以 ImageNet 均值/方差标准化后的 RGB 图像作为输入。
        // 这里的标准化保证感知损失的特征尺度和 VGG 训练时一致。
        x = (x - mean_) / std_;
    }
    return features_->forward(x);
}

void VGGFeatureExtractor::load_pretrained(const std::string& weight_path) {
    try {
        // 项目使用 LibTorch 自建 VGG 特征提取器。加载 TorchScript 权重时，
        // 按顺序把 JIT 模型里的卷积参数拷贝到当前 features_，避免依赖 Python 运行时。
        // 加载 TorchScript 模型，按顺序提取卷积层权重到自建的 features_ 中
        auto jit_module = torch::jit::load(weight_path);
        jit_module.eval();

        // 收集 JIT 模型中有参数的层（按顺序）
        std::vector<std::pair<torch::Tensor, torch::Tensor>> jit_conv_params;
        for (const auto& p : jit_module.named_parameters()) {
            // weight 和 bias 成对出现，先收集 weight
            if (p.name.find("weight") != std::string::npos) {
                jit_conv_params.push_back({p.value, {}});
            } else if (p.name.find("bias") != std::string::npos && !jit_conv_params.empty()) {
                jit_conv_params.back().second = p.value;
            }
        }

        // 收集自建 features_ 中的 Conv2d 层（按顺序）
        std::vector<torch::nn::Conv2dImpl*> my_convs;
        for (size_t i = 0; i < features_->size(); ++i) {
            auto conv = features_[i]->as<torch::nn::Conv2dImpl>();
            if (conv) {
                my_convs.push_back(conv);
            }
        }

        int loaded = 0;
        size_t count = std::min(my_convs.size(), jit_conv_params.size());
        for (size_t i = 0; i < count; ++i) {
            if (my_convs[i]->weight.sizes() == jit_conv_params[i].first.sizes()) {
                my_convs[i]->weight.data().copy_(jit_conv_params[i].first.data());
                if (jit_conv_params[i].second.defined() && my_convs[i]->bias.defined()) {
                    my_convs[i]->bias.data().copy_(jit_conv_params[i].second.data());
                }
                loaded++;
            }
        }

        // 冻结参数
        for (auto& param : parameters()) {
            param.set_requires_grad(false);
        }
        LOG_INFO("Loaded VGG pretrained weights from: ", weight_path,
                 " (", loaded, "/", my_convs.size(), " conv layers matched)");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load VGG weights: ", e.what());
        throw;
    }
}


// ==================== PerceptualLoss ====================

PerceptualLoss::PerceptualLoss(int feature_layer, PixelLossType loss_type)
    : loss_type_(loss_type) {
    vgg_ = std::make_shared<VGGFeatureExtractor>(feature_layer, true);
    register_module("vgg", vgg_);
}

PerceptualLoss::PerceptualLoss(int feature_layer, const std::string& loss_type)
    : PerceptualLoss(feature_layer, stringToPixelLossType(loss_type)) {
}

torch::Tensor PerceptualLoss::forward(torch::Tensor pred, torch::Tensor target) {
    auto pred_feat = vgg_->forward(pred);
    auto target_feat = vgg_->forward(target);

    // 感知损失比较的是特征图差异。它允许像素有轻微偏移，
    // 但会惩罚纹理、边缘和语义结构在深层特征空间中的不一致。
    switch (loss_type_) {
        case PixelLossType::L1:
            return F::l1_loss(pred_feat, target_feat);
        case PixelLossType::L2:
            return F::mse_loss(pred_feat, target_feat);
        default:
            return F::l1_loss(pred_feat, target_feat);
    }
}

void PerceptualLoss::load_vgg_weights(const std::string& weight_path) {
    vgg_->load_pretrained(weight_path);
}


// ==================== GANLoss ====================

GANLoss::GANLoss(GANType gan_type, double real_label_val, double fake_label_val)
    : gan_type_(gan_type), real_label_val_(real_label_val), fake_label_val_(fake_label_val) {
    LOG_DEBUG("GANLoss created with type: ", ganTypeToString(gan_type));
}

GANLoss::GANLoss(const std::string& gan_type, double real_label_val, double fake_label_val)
    : GANLoss(stringToGANType(gan_type), real_label_val, fake_label_val) {
}

torch::Tensor GANLoss::getTargetLabel(torch::Tensor input, bool target_is_real) {
    double label_val = target_is_real ? real_label_val_ : fake_label_val_;
    return torch::full_like(input, label_val);
}

torch::Tensor GANLoss::computeVanillaLoss(torch::Tensor input, torch::Tensor target) {
    return F::binary_cross_entropy_with_logits(input, target);
}

torch::Tensor GANLoss::computeLSGANLoss(torch::Tensor input, torch::Tensor target) {
    return F::mse_loss(input, target);
}

torch::Tensor GANLoss::computeWGANLoss(torch::Tensor input, bool target_is_real) {
    // WGAN 的目标直接使用判别器/critic 分数均值。
    // 这里没有实现 WGAN-GP 的梯度惩罚项，GANType::WGAN_GP 会和 WGAN 共用该计算路径。
    return target_is_real ? -input.mean() : input.mean();
}

torch::Tensor GANLoss::computeHingeLoss(torch::Tensor input, bool target_is_real, bool is_disc) {
    if (is_disc) {
        if (target_is_real) {
            return F::relu(1.0 - input).mean();
        } else {
            return F::relu(1.0 + input).mean();
        }
    } else {
        return -input.mean();
    }
}

torch::Tensor GANLoss::forward(torch::Tensor input, bool target_is_real, bool is_disc) {
    switch (gan_type_) {
        case GANType::Vanilla: {
            auto target = getTargetLabel(input, target_is_real);
            return computeVanillaLoss(input, target);
        }
        case GANType::LSGAN: {
            auto target = getTargetLabel(input, target_is_real);
            return computeLSGANLoss(input, target);
        }
        case GANType::WGAN:
        case GANType::WGAN_GP:
            return computeWGANLoss(input, target_is_real);

        case GANType::Hinge:
            return computeHingeLoss(input, target_is_real, is_disc);

        default:
            TORCH_CHECK(false, "Unsupported GAN type");
            return {};
    }
}


// ==================== CombinedLoss ====================

CombinedLoss::CombinedLoss(double pixel_weight, double perceptual_weight, double gan_weight,
                           PixelLossType pixel_loss_type, GANType gan_type)
    : pixel_weight_(pixel_weight),
      perceptual_weight_(perceptual_weight),
      gan_weight_(gan_weight) {

    pixel_loss_ = std::make_shared<PixelLoss>(pixel_loss_type);
    perceptual_loss_ = std::make_shared<PerceptualLoss>(
        constants::DEFAULT_VGG_FEATURE_LAYER, PixelLossType::L1);
    gan_loss_ = std::make_shared<GANLoss>(gan_type);

    register_module("pixel_loss", pixel_loss_);
    register_module("perceptual_loss", perceptual_loss_);
    register_module("gan_loss", gan_loss_);

    LOG_DEBUG("CombinedLoss created with weights: pixel=", pixel_weight,
              ", perceptual=", perceptual_weight, ", gan=", gan_weight);
}

std::map<std::string, torch::Tensor> CombinedLoss::forward(
    torch::Tensor pred,
    torch::Tensor target,
    torch::Tensor disc_pred_fake,
    bool use_perceptual,
    bool use_gan) {

    std::map<std::string, torch::Tensor> losses;
    auto device = pred.device();

    // 三阶段训练的核心就在这里：
    // - use_perceptual=false, use_gan=false：只返回像素损失；
    // - use_perceptual=true,  use_gan=false：加入 VGG 感知损失；
    // - use_perceptual=true,  use_gan=true ：再加入生成器的对抗损失。
    // 像素损失
    losses["pixel"] = pixel_loss_->forward(pred, target) * pixel_weight_;

    // 感知损失
    if (use_perceptual) {
        losses["perceptual"] = perceptual_loss_->forward(pred, target) * perceptual_weight_;
    } else {
        losses["perceptual"] = torch::zeros({1}, torch::TensorOptions().device(device));
    }

    // GAN损失
    if (use_gan && disc_pred_fake.defined()) {
        losses["gan"] = gan_loss_->forward(disc_pred_fake, true) * gan_weight_;
    } else {
        losses["gan"] = torch::zeros({1}, torch::TensorOptions().device(device));
    }

    // 总损失
    losses["total"] = losses["pixel"] + losses["perceptual"] + losses["gan"];

    return losses;
}

void CombinedLoss::load_vgg_weights(const std::string& weight_path) {
    perceptual_loss_->load_vgg_weights(weight_path);
}

}  // namespace models
}  // namespace facesr
