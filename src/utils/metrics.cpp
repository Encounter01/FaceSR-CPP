/**
 * @file metrics.cpp
 * @brief 评估指标模块实现
 */

#include "utils/metrics.h"
#include <cmath>

namespace facesr {
namespace utils {

double calculate_psnr(torch::Tensor pred, torch::Tensor target, double max_val) {
    // 确保在CPU上
    pred = pred.detach().cpu();
    target = target.detach().cpu();

    // 裁剪到[0, max_val]
    pred = pred.clamp(0, max_val);
    target = target.clamp(0, max_val);

    // 计算MSE
    auto diff = pred - target;
    double mse = diff.pow(2).mean().item<double>();

    if (mse == 0) {
        return std::numeric_limits<double>::infinity();
    }

    // 计算PSNR
    double psnr = 20.0 * std::log10(max_val / std::sqrt(mse));

    return psnr;
}

double calculate_ssim(torch::Tensor pred, torch::Tensor target, int window_size) {
    // 确保在CPU上
    pred = pred.detach().cpu().to(torch::kFloat32);
    target = target.detach().cpu().to(torch::kFloat32);

    // 确保4维
    if (pred.dim() == 3) {
        pred = pred.unsqueeze(0);
    }
    if (target.dim() == 3) {
        target = target.unsqueeze(0);
    }

    int channels = pred.size(1);

    // 创建高斯窗口
    double sigma = 1.5;
    std::vector<double> gauss_values;
    for (int i = 0; i < window_size; ++i) {
        double val = std::exp(-std::pow(i - window_size / 2, 2) / (2 * sigma * sigma));
        gauss_values.push_back(val);
    }

    // 归一化
    double sum = 0;
    for (double v : gauss_values) sum += v;
    for (double& v : gauss_values) v /= sum;

    // 创建2D窗口
    auto window_1d = torch::tensor(gauss_values, torch::kFloat32).unsqueeze(1);
    auto window_2d = torch::mm(window_1d, window_1d.t()).unsqueeze(0).unsqueeze(0);
    auto window = window_2d.expand({channels, 1, window_size, window_size}).contiguous();

    // SSIM参数
    double C1 = std::pow(0.01, 2);
    double C2 = std::pow(0.03, 2);

    int padding = window_size / 2;

    // 计算均值
    auto mu1 = torch::conv2d(pred, window, {}, 1, padding, 1, channels);
    auto mu2 = torch::conv2d(target, window, {}, 1, padding, 1, channels);

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    // 计算方差和协方差
    auto sigma1_sq = torch::conv2d(pred.pow(2), window, {}, 1, padding, 1, channels) - mu1_sq;
    auto sigma2_sq = torch::conv2d(target.pow(2), window, {}, 1, padding, 1, channels) - mu2_sq;
    auto sigma12 = torch::conv2d(pred * target, window, {}, 1, padding, 1, channels) - mu1_mu2;

    // 计算SSIM
    auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    return ssim_map.mean().item<double>();
}

// ==================== MetricCalculator ====================

void MetricCalculator::reset() {
    psnr_sum_ = 0;
    ssim_sum_ = 0;
    count_ = 0;
}

void MetricCalculator::update(torch::Tensor pred, torch::Tensor target) {
    int batch_size = pred.size(0);

    for (int i = 0; i < batch_size; ++i) {
        auto pred_i = pred[i].unsqueeze(0);
        auto target_i = target[i].unsqueeze(0);

        psnr_sum_ += calculate_psnr(pred_i, target_i);
        ssim_sum_ += calculate_ssim(pred_i, target_i);
        count_++;
    }
}

double MetricCalculator::get_avg_psnr() const {
    return count_ > 0 ? psnr_sum_ / count_ : 0;
}

double MetricCalculator::get_avg_ssim() const {
    return count_ > 0 ? ssim_sum_ / count_ : 0;
}


// ==================== AverageMeter ====================

void AverageMeter::reset() {
    val_ = 0;
    avg_ = 0;
    sum_ = 0;
    count_ = 0;
}

void AverageMeter::update(double val, int n) {
    val_ = val;
    sum_ += val * n;
    count_ += n;
    avg_ = sum_ / count_;
}

}  // namespace utils
}  // namespace facesr
