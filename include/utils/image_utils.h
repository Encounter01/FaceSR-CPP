#pragma once
/**
 * @file image_utils.h
 * @brief 图像处理工具
 */

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace facesr {
namespace utils {

/**
 * @brief Tensor转换为OpenCV Mat
 * @param tensor 输入Tensor [C, H, W] 或 [B, C, H, W]
 * @return OpenCV Mat (BGR, uint8)
 */
cv::Mat tensor_to_mat(torch::Tensor tensor);

/**
 * @brief OpenCV Mat转换为Tensor
 * @param img OpenCV Mat (BGR 或 RGB)
 * @param is_bgr 输入是否为BGR格式
 * @return Tensor [1, C, H, W], 值范围[0, 1]
 */
torch::Tensor mat_to_tensor(const cv::Mat& img, bool is_bgr = true);

/**
 * @brief 保存Tensor为图像
 * @param tensor 输入Tensor
 * @param path 保存路径
 */
void save_tensor_image(torch::Tensor tensor, const std::string& path);

/**
 * @brief 加载图像为Tensor
 * @param path 图像路径
 * @param target_size 目标尺寸 (可选)
 * @return Tensor [1, C, H, W]
 */
torch::Tensor load_image_tensor(const std::string& path,
                                 int target_size = -1);

/**
 * @brief 创建对比图像
 * @param lr 低分辨率图像
 * @param sr 超分辨率图像
 * @param hr 高分辨率真值 (可选)
 * @return 对比图像
 */
cv::Mat create_comparison_image(torch::Tensor lr, torch::Tensor sr,
                                 torch::Tensor hr = {});

}  // namespace utils
}  // namespace facesr
