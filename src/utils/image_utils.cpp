/**
 * @file image_utils.cpp
 * @brief 图像处理工具实现
 */

#include "utils/image_utils.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace facesr {
namespace utils {

cv::Mat tensor_to_mat(torch::Tensor tensor) {
    // 确保在CPU上
    tensor = tensor.detach().cpu();

    // 处理batch维度
    if (tensor.dim() == 4) {
        tensor = tensor[0];
    }

    // 裁剪到[0, 1]
    tensor = tensor.clamp(0, 1);

    // [C, H, W] -> [H, W, C]
    tensor = tensor.permute({1, 2, 0});

    // 转换为uint8
    tensor = (tensor * 255).to(torch::kUInt8);

    // 确保连续
    tensor = tensor.contiguous();

    // 创建Mat
    int h = tensor.size(0);
    int w = tensor.size(1);
    cv::Mat img(h, w, CV_8UC3, tensor.data_ptr<uint8_t>());

    // 克隆以拥有数据所有权
    cv::Mat result = img.clone();

    // RGB to BGR
    cv::cvtColor(result, result, cv::COLOR_RGB2BGR);

    return result;
}

torch::Tensor mat_to_tensor(const cv::Mat& img, bool is_bgr) {
    cv::Mat rgb_img;
    if (is_bgr) {
        cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    } else {
        rgb_img = img.clone();
    }

    // 转换为float
    cv::Mat float_img;
    rgb_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 创建Tensor
    auto tensor = torch::from_blob(
        float_img.data,
        {1, float_img.rows, float_img.cols, 3},
        torch::kFloat32
    ).clone();

    // [B, H, W, C] -> [B, C, H, W]
    tensor = tensor.permute({0, 3, 1, 2});

    return tensor;
}

void save_tensor_image(torch::Tensor tensor, const std::string& path) {
    // 确保目录存在
    fs::path file_path(path);
    fs::create_directories(file_path.parent_path());

    // 转换为Mat并保存
    cv::Mat img = tensor_to_mat(tensor);
    cv::imwrite(path, img);
}

torch::Tensor load_image_tensor(const std::string& path, int target_size) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    // 调整尺寸
    if (target_size > 0) {
        cv::resize(img, img, cv::Size(target_size, target_size), 0, 0, cv::INTER_CUBIC);
    }

    return mat_to_tensor(img, true);
}

cv::Mat create_comparison_image(torch::Tensor lr, torch::Tensor sr, torch::Tensor hr) {
    // 转换为Mat
    cv::Mat lr_mat = tensor_to_mat(lr);
    cv::Mat sr_mat = tensor_to_mat(sr);

    // 将LR放大到与SR相同尺寸
    cv::resize(lr_mat, lr_mat, sr_mat.size(), 0, 0, cv::INTER_NEAREST);

    std::vector<cv::Mat> images = {lr_mat, sr_mat};
    std::vector<std::string> labels = {"LR (Input)", "SR (Output)"};

    if (hr.defined() && hr.numel() > 0) {
        cv::Mat hr_mat = tensor_to_mat(hr);
        images.push_back(hr_mat);
        labels.push_back("HR (Ground Truth)");
    }

    // 计算总尺寸
    int h = sr_mat.rows;
    int w = sr_mat.cols;
    int padding = 10;
    int total_width = w * images.size() + padding * (images.size() - 1);

    // 创建对比图 (加上标签空间)
    cv::Mat comparison(h + 30, total_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 放置图像和标签
    for (size_t i = 0; i < images.size(); ++i) {
        int x_offset = i * (w + padding);

        // 复制图像
        images[i].copyTo(comparison(cv::Rect(x_offset, 0, w, h)));

        // 添加标签
        cv::putText(comparison, labels[i],
                    cv::Point(x_offset + 5, h + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
    }

    return comparison;
}

}  // namespace utils
}  // namespace facesr
