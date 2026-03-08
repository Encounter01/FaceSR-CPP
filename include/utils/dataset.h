#pragma once
/**
 * @file dataset.h
 * @brief 数据集加载模块
 *
 * 用于加载和预处理人脸图像数据
 */

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>
#include "common/config.h"

namespace facesr {
namespace utils {

namespace fs = std::filesystem;

/**
 * @brief 数据增强配置
 */
struct AugmentationConfig {
    bool enable_flip = true;
    bool enable_rotation = true;
    double flip_probability = constants::AUGMENT_FLIP_PROB;
    double rotation_probability = constants::AUGMENT_ROTATE_PROB;
};

/**
 * @brief 人脸超分辨率数据集
 */
class FaceSRDataset : public torch::data::Dataset<FaceSRDataset> {
public:
    /**
     * @param hr_dir 高分辨率图像目录
     * @param lr_dir 低分辨率图像目录 (可选)
     * @param hr_size 高分辨率图像尺寸
     * @param lr_size 低分辨率图像尺寸
     * @param augment 是否进行数据增强
     */
    FaceSRDataset(const std::string& hr_dir,
                  const std::string& lr_dir = "",
                  int hr_size = constants::DEFAULT_HR_SIZE,
                  int lr_size = constants::DEFAULT_LR_SIZE,
                  bool augment = true);

    /**
     * @brief 带增强配置的构造函数
     */
    FaceSRDataset(const std::string& hr_dir,
                  const std::string& lr_dir,
                  int hr_size,
                  int lr_size,
                  const AugmentationConfig& aug_config);

    /**
     * @brief 获取单个样本
     */
    torch::data::Example<> get(size_t index) override;

    /**
     * @brief 获取数据集大小
     */
    torch::optional<size_t> size() const override;

    /**
     * @brief 获取文件名
     */
    std::string get_filename(size_t index) const;

    /**
     * @brief 检查数据集是否为空
     */
    bool empty() const { return hr_images_.empty(); }

    /**
     * @brief 设置数据增强配置
     */
    void setAugmentationConfig(const AugmentationConfig& config) {
        aug_config_ = config;
    }

private:
    /**
     * @brief 获取目录下所有图像文件
     */
    std::vector<fs::path> getImageFiles(const std::string& dir);

    /**
     * @brief 检查是否为支持的图像格式
     */
    bool isSupportedImageFormat(const std::string& extension) const;

    /**
     * @brief 加载图像
     */
    cv::Mat loadImage(const fs::path& path);

    /**
     * @brief 应用数据增强
     */
    std::pair<cv::Mat, cv::Mat> applyAugmentation(cv::Mat hr, cv::Mat lr);

    /**
     * @brief 将OpenCV Mat转换为Tensor
     */
    torch::Tensor matToTensor(const cv::Mat& img);

    std::vector<fs::path> hr_images_;
    std::string lr_dir_;
    int hr_size_;
    int lr_size_;
    AugmentationConfig aug_config_;
    bool augment_;
};


/**
 * @brief 数据加载器配置
 */
struct DataLoaderConfig {
    int batch_size = constants::DEFAULT_BATCH_SIZE;
    int num_workers = constants::DEFAULT_NUM_WORKERS;
    bool shuffle = true;
    bool drop_last = false;
};

/**
 * @brief 创建数据加载器选项
 */
torch::data::DataLoaderOptions createDataLoaderOptions(const DataLoaderConfig& config);

/**
 * @brief 创建数据加载器选项（兼容旧API）
 */
torch::data::DataLoaderOptions get_dataloader_options(int batch_size, int num_workers, bool shuffle);

}  // namespace utils
}  // namespace facesr
