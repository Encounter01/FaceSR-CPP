// ============================================================================
// inference.cpp — 推理器实现
//
// 推理（inference）是模型训练完成后的应用阶段:
// - 不需要梯度计算（NoGradGuard）
// - 不需要判别器和损失函数
// - 使用eval()模式（BatchNorm使用训练时的滑动统计量）
// ============================================================================

#include "inference.h"
#include "utils/image_utils.h"
#include "common/config.h"
#include "common/logger.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace facesr {

// ============================================================================
// 构造函数 — 加载模型并初始化
// ============================================================================
Inference::Inference(const std::string& model_path, int scale,
                     bool use_gpu, bool use_attention)
    : device_(torch::kCPU)  // 先初始化为CPU
    , scale_(scale) {

    // 选择推理设备: GPU可用且用户要求时使用GPU
    if (use_gpu && torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA);
        LOG_INFO("Inference using CUDA");
    } else {
        LOG_INFO("Inference using CPU");
    }

    // 创建RRDB生成器（参数与训练时必须一致）
    generator_ = RRDBNet(3, 3, constants::RRDB_NUM_FEAT,
                         constants::RRDB_NUM_BLOCK,
                         constants::RRDB_NUM_GROW_CH, scale,
                         use_attention);
    generator_->to(device_);  // 移到推理设备

    // 加载训练好的权重
    try {
        torch::load(generator_, model_path);
        generator_->eval();  // 设置为评估模式（关闭Dropout，BN使用滑动统计量）
        model_loaded_ = true;
        LOG_INFO("Model loaded successfully from: ", model_path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load model: ", e.what());
        model_loaded_ = false;
    }
}

// ============================================================================
// process — 处理单张图像（核心推理函数）
//
// 完整推理流程:
// cv::Mat(BGR) → mat_to_tensor(RGB float) → Generator → clamp → tensor_to_mat(BGR)
// ============================================================================
cv::Mat Inference::process(const cv::Mat& input) {
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded!");
        return {};
    }

    // Mat → Tensor: BGR uint8 → RGB float [1,3,H,W], 值域[0,1]
    auto input_tensor = image_utils::mat_to_tensor(input);
    input_tensor = input_tensor.to(device_);  // 移到推理设备

    // 前向传播（禁用梯度计算，节省内存和加速）
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;  // RAII式禁用梯度
        output = generator_->forward(input_tensor);
        output = output.clamp(0.0, 1.0);  // clamp到[0,1]防止越界
    }

    // Tensor → Mat: RGB float → BGR uint8
    return image_utils::tensor_to_mat(output.cpu());
}

// ============================================================================
// process_file — 处理单个图像文件
// ============================================================================
bool Inference::process_file(const std::string& input_path,
                             const std::string& output_path) {
    // 读取输入图像
    cv::Mat input = cv::imread(input_path, cv::IMREAD_COLOR);
    if (input.empty()) {
        LOG_ERROR("Failed to read image: ", input_path);
        return false;
    }

    LOG_INFO("Processing: ", input_path,
             " (", input.cols, "x", input.rows, ")");

    // 执行超分辨率推理
    cv::Mat result = process(input);
    if (result.empty()) {
        LOG_ERROR("Processing failed for: ", input_path);
        return false;
    }

    // 保存结果
    bool success = cv::imwrite(output_path, result);
    if (success) {
        LOG_INFO("Saved: ", output_path,
                 " (", result.cols, "x", result.rows, ")");
    } else {
        LOG_ERROR("Failed to save: ", output_path);
    }
    return success;
}

// ============================================================================
// process_folder — 批量处理文件夹中的所有图像
// 返回: 成功处理的图像数量
// ============================================================================
int Inference::process_folder(const std::string& input_dir,
                              const std::string& output_dir) {
    // 创建输出目录
    fs::create_directories(output_dir);

    int processed = 0;
    int failed = 0;

    // 遍历目录中的所有图像文件
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // 检查是否为支持的图像格式
        bool supported = false;
        for (const auto& fmt : constants::SUPPORTED_FORMATS) {
            if (ext == fmt) { supported = true; break; }
        }
        if (!supported) continue;

        // 构建输出路径: 原始文件名 + "_sr" + 扩展名
        std::string filename = entry.path().stem().string();
        std::string output_path = output_dir + "/" + filename + "_sr" + ext;

        // 处理单张图像
        try {
            if (process_file(entry.path().string(), output_path)) {
                ++processed;
            } else {
                ++failed;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Error processing ", entry.path().string(), ": ", e.what());
            ++failed;
        }
    }

    LOG_INFO("Batch processing complete: ", processed, " succeeded, ", failed, " failed");
    return processed;
}

}  // namespace facesr
