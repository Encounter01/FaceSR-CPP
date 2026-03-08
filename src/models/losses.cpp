/**
 * @file losses.cpp
 * @brief 损失函数模块实现
 */

#include "models/losses.h"
#include "common/logger.h"
#include <torch/nn/functional.h>

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
        x = (x - mean_) / std_;
    }
    return features_->forward(x);
}

void VGGFeatureExtractor::load_pretrained(const std::string& weight_path) {
    try {
        torch::load(features_, weight_path);
        // 冻结参数
        for (auto& param : parameters()) {
            param.set_requires_grad(false);
        }
        LOG_INFO("Loaded VGG pretrained weights from: ", weight_path);
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
