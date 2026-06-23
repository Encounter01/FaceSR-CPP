#pragma once
/**
 * @file metrics.h
 * @brief 评估指标模块
 *
 * 论文正式定量结果主要依赖 PSNR 和 SSIM：
 * - PSNR 关注像素误差，能反映 SR 与 HR 的数值接近程度。
 * - SSIM 关注亮度、对比度和结构，更适合补充说明人脸轮廓与纹理结构保持情况。
 * GAN 结果的视觉观感不一定和 PSNR 完全一致，因此论文中还结合可视化面板分析。
 */

#include <torch/torch.h>

namespace facesr {
namespace utils {

/**
 * @brief 计算PSNR (峰值信噪比)
 * @param pred 预测图像
 * @param target 目标图像
 * @param max_val 像素最大值
 * @return PSNR值 (dB)
 */
double calculate_psnr(torch::Tensor pred, torch::Tensor target, double max_val = 1.0);

/**
 * @brief 计算SSIM (结构相似性)
 * @param pred 预测图像
 * @param target 目标图像
 * @param window_size 窗口大小
 * @return SSIM值
 */
double calculate_ssim(torch::Tensor pred, torch::Tensor target, int window_size = 11);

/**
 * @brief 指标计算器类
 */
class MetricCalculator {
public:
    MetricCalculator() { reset(); }

    /**
     * @brief 重置累积指标
     */
    void reset();

    /**
     * @brief 更新指标
     */
    void update(torch::Tensor pred, torch::Tensor target);

    /**
     * @brief 获取平均PSNR
     */
    double get_avg_psnr() const;

    /**
     * @brief 获取平均SSIM
     */
    double get_avg_ssim() const;

    /**
     * @brief 获取样本数量
     */
    int get_count() const { return count_; }

private:
    double psnr_sum_;
    double ssim_sum_;
    int count_;
};

/**
 * @brief 平均值记录器
 */
class AverageMeter {
public:
    AverageMeter() { reset(); }

    void reset();
    void update(double val, int n = 1);

    double val() const { return val_; }
    double avg() const { return avg_; }
    double sum() const { return sum_; }
    int count() const { return count_; }

private:
    double val_;
    double avg_;
    double sum_;
    int count_;
};

}  // namespace utils
}  // namespace facesr
