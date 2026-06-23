/**
 * @file dataset.cpp
 * @brief 数据集加载模块实现
 *
 * 本文件是训练数据入口。读代码时重点关注 get()：
 * 它决定一个样本如何从磁盘图像变成训练器里的 LR/HR 张量对。
 */

#include "utils/dataset.h"
#include "common/random.h"
#include "common/logger.h"
#include <algorithm>

namespace facesr {
namespace utils {

FaceSRDataset::FaceSRDataset(const std::string& hr_dir,
                             const std::string& lr_dir,
                             int hr_size,
                             int lr_size,
                             bool augment)
    : lr_dir_(lr_dir), hr_size_(hr_size), lr_size_(lr_size), augment_(augment) {

    // 设置默认增强配置
    aug_config_.enable_flip = augment;
    aug_config_.enable_rotation = augment;

    hr_images_ = getImageFiles(hr_dir);

    if (hr_images_.empty()) {
        LOG_WARN("No images found in ", hr_dir);
    } else {
        LOG_INFO("Loaded ", hr_images_.size(), " images from ", hr_dir);
    }
}

FaceSRDataset::FaceSRDataset(const std::string& hr_dir,
                             const std::string& lr_dir,
                             int hr_size,
                             int lr_size,
                             const AugmentationConfig& aug_config)
    : lr_dir_(lr_dir), hr_size_(hr_size), lr_size_(lr_size),
      aug_config_(aug_config), augment_(aug_config.enable_flip || aug_config.enable_rotation) {

    hr_images_ = getImageFiles(hr_dir);

    if (hr_images_.empty()) {
        LOG_WARN("No images found in ", hr_dir);
    } else {
        LOG_INFO("Loaded ", hr_images_.size(), " images from ", hr_dir);
    }
}

bool FaceSRDataset::isSupportedImageFormat(const std::string& extension) const {
    std::string ext_lower = extension;
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);

    for (size_t i = 0; i < constants::NUM_SUPPORTED_EXTENSIONS; ++i) {
        if (ext_lower == constants::SUPPORTED_EXTENSIONS[i]) {
            return true;
        }
    }
    return false;
}

std::vector<fs::path> FaceSRDataset::getImageFiles(const std::string& dir) {
    std::vector<fs::path> files;

    if (!fs::exists(dir)) {
        LOG_ERROR("Directory does not exist: ", dir);
        return files;
    }

    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (isSupportedImageFormat(ext)) {
                    files.push_back(entry.path());
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Error reading directory ", dir, ": ", e.what());
        return files;
    }

    // 排序以保证顺序一致
    std::sort(files.begin(), files.end());
    return files;
}

cv::Mat FaceSRDataset::loadImage(const fs::path& path) {
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + path.string());
    }
    // OpenCV 读入默认是 BGR，而模型训练统一使用 RGB。
    // 这个转换必须和 image_utils 中的推理转换保持一致，否则颜色通道会错位。
    // BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return img;
}

std::pair<cv::Mat, cv::Mat> FaceSRDataset::applyAugmentation(cv::Mat hr, cv::Mat lr) {
    // 增强必须同时作用在 HR 和 LR 上，否则输入与标签会空间错位。
    // 人脸超分辨率任务对眼睛、鼻子、嘴部结构位置敏感，因此这里不能分别随机增强。
    // 使用线程安全的随机数生成器
    auto& rng = RandomGenerator::getInstance();

    // 随机水平翻转
    if (aug_config_.enable_flip && rng.bernoulli(aug_config_.flip_probability)) {
        cv::flip(hr, hr, 1);
        cv::flip(lr, lr, 1);
    }

    // 随机90度旋转
    if (aug_config_.enable_rotation && rng.bernoulli(aug_config_.rotation_probability)) {
        int rotations = rng.uniformInt(1, 3);  // 1, 2, or 3 times 90 degrees
        for (int i = 0; i < rotations; ++i) {
            cv::rotate(hr, hr, cv::ROTATE_90_CLOCKWISE);
            cv::rotate(lr, lr, cv::ROTATE_90_CLOCKWISE);
        }
    }

    return {hr, lr};
}

torch::Tensor FaceSRDataset::matToTensor(const cv::Mat& img) {
    // Dataset 内部的 Mat 已经是 RGB，因此这里只做数值归一化和 HWC->CHW 维度转换。
    // clone() 很重要：from_blob 不拥有 OpenCV Mat 的内存，离开函数后必须复制出独立 Tensor。
    // 转换为float并归一化到[0, 1]
    cv::Mat float_img;
    img.convertTo(float_img, CV_32FC3, constants::NORMALIZE_SCALE);

    // 创建Tensor [H, W, C] -> [C, H, W]
    auto tensor = torch::from_blob(
        float_img.data,
        {float_img.rows, float_img.cols, 3},
        torch::kFloat32
    ).clone();

    // 转换为 [C, H, W]
    tensor = tensor.permute({2, 0, 1});

    return tensor;
}

torch::data::Example<> FaceSRDataset::get(size_t index) {
    // 读取并构造成一个训练样本。
    // 论文主线是受控 Bicubic 退化：如果没有单独 LR 目录，就直接从 HR 下采样得到 LR。
    // 加载HR图像
    cv::Mat hr_img = loadImage(hr_images_[index]);

    // 调整HR图像尺寸
    cv::resize(hr_img, hr_img, cv::Size(hr_size_, hr_size_), 0, 0, cv::INTER_CUBIC);

    // 加载或生成 LR 图像。
    // 有 LR 目录时优先读取同名文件；没有或同名文件不存在时，回退到 HR 的 Bicubic 下采样。
    // 加载或生成LR图像
    cv::Mat lr_img;
    if (!lr_dir_.empty()) {
        fs::path lr_path = fs::path(lr_dir_) / hr_images_[index].filename();
        if (fs::exists(lr_path)) {
            lr_img = loadImage(lr_path);
            cv::resize(lr_img, lr_img, cv::Size(lr_size_, lr_size_), 0, 0, cv::INTER_CUBIC);
        } else {
            // LR不存在，在线生成
            cv::resize(hr_img, lr_img, cv::Size(lr_size_, lr_size_), 0, 0, cv::INTER_CUBIC);
        }
    } else {
        // 在线生成LR图像
        cv::resize(hr_img, lr_img, cv::Size(lr_size_, lr_size_), 0, 0, cv::INTER_CUBIC);
    }

    // 数据增强
    if (augment_) {
        std::tie(hr_img, lr_img) = applyAugmentation(hr_img, lr_img);
    }

    // 转换为Tensor
    torch::Tensor hr_tensor = matToTensor(hr_img);
    torch::Tensor lr_tensor = matToTensor(lr_img);

    return {lr_tensor, hr_tensor};
}

torch::optional<size_t> FaceSRDataset::size() const {
    return hr_images_.size();
}

std::string FaceSRDataset::get_filename(size_t index) const {
    return hr_images_[index].stem().string();
}

torch::data::DataLoaderOptions createDataLoaderOptions(const DataLoaderConfig& config) {
    return torch::data::DataLoaderOptions()
        .batch_size(config.batch_size)
        .workers(config.num_workers)
        .enforce_ordering(!config.shuffle)
        .drop_last(config.drop_last);
}

torch::data::DataLoaderOptions get_dataloader_options(int batch_size, int num_workers, bool shuffle) {
    DataLoaderConfig config;
    config.batch_size = batch_size;
    config.num_workers = num_workers;
    config.shuffle = shuffle;
    return createDataLoaderOptions(config);
}

}  // namespace utils
}  // namespace facesr
