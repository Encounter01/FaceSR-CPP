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
// load_model_from_file — 尝试加载模型文件
// 先尝试 TorchScript (jit::load)，失败则尝试 torch::load
// ============================================================================
static bool try_load_jit(const std::string& path, torch::jit::Module& out,
                         torch::Device device) {
    try {
        out = torch::jit::load(path, device);
        out.eval();
        return true;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// 构造函数 — 加载模型并初始化
// ============================================================================
Inference::Inference(const std::string& model_path, int scale,
                     bool use_gpu, bool use_attention)
    : device_(torch::kCPU)
    , scale_(scale) {

    if (use_gpu && torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA);
        LOG_INFO("Inference using CUDA");
    } else {
        LOG_INFO("Inference using CPU");
    }

    // 先尝试 TorchScript 格式 (Python torch.jit.save 保存的)
    if (try_load_jit(model_path, jit_module_, device_)) {
        use_jit_ = true;
        model_loaded_ = true;
        LOG_INFO("Loaded TorchScript model from: ", model_path);
        return;
    }

    // 回退到 LibTorch 原生格式 (C++ torch::save 保存的)
    generator_ = RRDBNet(3, 3, constants::RRDB_NUM_FEATURES,
                         constants::RRDB_NUM_BLOCKS,
                         constants::RRDB_GROWTH_CHANNELS, scale,
                         use_attention);
    generator_->to(device_);
    try {
        torch::load(generator_, model_path);
        generator_->eval();
        model_loaded_ = true;
        LOG_INFO("Loaded LibTorch model from: ", model_path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load model: ", e.what());
        model_loaded_ = false;
    }
}

Inference::Inference(const std::string& model_path, int scale, DeviceType device_type)
    : device_(torch::kCPU)
    , scale_(scale)
    , device_type_(device_type) {

    bool use_gpu = (device_type == DeviceType::CUDA || device_type == DeviceType::Auto ||
                    device_type == DeviceType::Hybrid);
    if (use_gpu && torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA);
        LOG_INFO("Inference using CUDA");
    } else {
        LOG_INFO("Inference using CPU");
    }

    if (try_load_jit(model_path, jit_module_, device_)) {
        use_jit_ = true;
        model_loaded_ = true;
        LOG_INFO("Loaded TorchScript model from: ", model_path);
        return;
    }

    generator_ = RRDBNet(3, 3, constants::RRDB_NUM_FEATURES,
                         constants::RRDB_NUM_BLOCKS,
                         constants::RRDB_GROWTH_CHANNELS, scale);
    generator_->to(device_);
    try {
        torch::load(generator_, model_path);
        generator_->eval();
        model_loaded_ = true;
        LOG_INFO("Loaded LibTorch model from: ", model_path);
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

    auto input_tensor = utils::mat_to_tensor(input);
    input_tensor = input_tensor.to(device_);

    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        if (use_jit_) {
            output = jit_module_.forward({input_tensor}).toTensor();
        } else {
            output = generator_->forward(input_tensor);
        }
        output = output.clamp(0.0, 1.0);
    }

    return utils::tensor_to_mat(output.cpu());
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
        for (const auto& fmt : constants::SUPPORTED_EXTENSIONS) {
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
