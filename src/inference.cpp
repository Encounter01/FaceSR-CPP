/**
 * @file inference.cpp
 * @brief 推理器类实现 (支持GPU+CPU混合流水线)
 *
 * 混合模式流水线架构:
 *   [CPU线程] 图像加载+预处理  →  队列1  →  [GPU线程] 模型推理  →  队列2  →  [CPU线程] 后处理+保存
 *
 * 这种架构使GPU和CPU同时工作:
 * - CPU负责IO密集型操作: 磁盘读写, 图像解码/编码, 格式转换
 * - GPU负责计算密集型操作: 神经网络前向传播
 * - 通过流水线并行, GPU不需要等待CPU完成IO即可处理下一批数据
 */

#include "inference.h"
#include "utils/image_utils.h"
#include <iostream>
#include <filesystem>
#include <thread>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;

namespace facesr {

// ==================== 构造函数 ====================

Inference::Inference(const std::string& model_path, int scale, bool use_gpu)
    : gpu_device_(torch::kCPU),
      cpu_device_(torch::kCPU),
      scale_(scale) {

    if (use_gpu && torch::cuda::is_available()) {
        device_type_ = DeviceType::Hybrid;  // 有GPU时默认启用混合模式
        gpu_device_ = torch::Device(torch::kCUDA);
    } else {
        device_type_ = DeviceType::CPU;
    }

    init_model(model_path);
}

Inference::Inference(const std::string& model_path, int scale, DeviceType device_type)
    : gpu_device_(torch::kCPU),
      cpu_device_(torch::kCPU),
      scale_(scale),
      device_type_(device_type) {

    // 根据设备类型配置
    switch (device_type_) {
        case DeviceType::CUDA:
            if (torch::cuda::is_available()) {
                gpu_device_ = torch::Device(torch::kCUDA);
            } else {
                std::cerr << "警告: CUDA不可用, 回退到CPU模式" << std::endl;
                device_type_ = DeviceType::CPU;
            }
            break;

        case DeviceType::Hybrid:
            if (torch::cuda::is_available()) {
                gpu_device_ = torch::Device(torch::kCUDA);
            } else {
                std::cerr << "警告: CUDA不可用, 混合模式需要GPU, 回退到CPU模式" << std::endl;
                device_type_ = DeviceType::CPU;
            }
            break;

        case DeviceType::Auto:
            if (torch::cuda::is_available()) {
                device_type_ = DeviceType::Hybrid;  // Auto模式优先选择混合模式
                gpu_device_ = torch::Device(torch::kCUDA);
            } else {
                device_type_ = DeviceType::CPU;
            }
            break;

        case DeviceType::CPU:
        default:
            break;
    }

    init_model(model_path);
}

void Inference::init_model(const std::string& model_path) {
    // 打印设备信息
    switch (device_type_) {
        case DeviceType::Hybrid:
            std::cout << "运行模式: GPU+CPU混合流水线" << std::endl;
            std::cout << "  GPU设备: CUDA (模型推理)" << std::endl;
            std::cout << "  CPU设备: CPU (预处理/后处理/IO)" << std::endl;
            std::cout << "  流水线队列深度: " << constants::DEFAULT_PIPELINE_QUEUE_SIZE << std::endl;
            std::cout << "  CPU预处理线程: " << constants::DEFAULT_CPU_PREPROCESS_THREADS << std::endl;
            std::cout << "  CPU后处理线程: " << constants::DEFAULT_CPU_POSTPROCESS_THREADS << std::endl;
            break;
        case DeviceType::CUDA:
            std::cout << "运行模式: GPU (CUDA)" << std::endl;
            break;
        case DeviceType::CPU:
            std::cout << "运行模式: CPU" << std::endl;
            break;
        default:
            break;
    }

    try {
        // 创建生成器
        generator_ = models::RRDBNet(3, 3, 64, 23, 32, scale_);

        // 模型放在GPU上 (混合模式和CUDA模式)
        if (device_type_ == DeviceType::Hybrid || device_type_ == DeviceType::CUDA) {
            generator_->to(gpu_device_);
        }

        // 加载权重
        torch::load(generator_, model_path);
        generator_->eval();

        model_loaded_ = true;
        std::cout << "模型加载成功: " << model_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "模型加载失败: " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

// ==================== 单张图像处理 ====================

cv::Mat Inference::process(const cv::Mat& input) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    // [CPU] 预处理: Mat → Tensor
    auto input_tensor = utils::mat_to_tensor(input, true);

    // 将张量传输到推理设备
    torch::Device infer_device = (device_type_ == DeviceType::Hybrid || device_type_ == DeviceType::CUDA)
                                  ? gpu_device_ : cpu_device_;

    // 使用 non_blocking 传输, CPU可以继续执行其他操作
    input_tensor = input_tensor.to(infer_device, /*non_blocking=*/true);

    // [GPU/CPU] 推理
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        output = generator_->forward(input_tensor);
        output = output.clamp(0, 1);
    }

    // [CPU] 后处理: 将结果传回CPU并转换为Mat
    if (output.device() != cpu_device_) {
        output = output.to(cpu_device_);
    }

    return utils::tensor_to_mat(output);
}

// ==================== 文件处理 ====================

void Inference::process_file(const std::string& input_path, const std::string& output_path) {
    // [CPU] 读取图像
    cv::Mat input = cv::imread(input_path, cv::IMREAD_COLOR);
    if (input.empty()) {
        throw std::runtime_error("无法加载图像: " + input_path);
    }

    std::cout << "处理: " << input_path << std::endl;
    std::cout << "  输入尺寸: " << input.cols << "x" << input.rows << std::endl;

    // 处理 (内部自动使用GPU或CPU)
    cv::Mat output = process(input);

    std::cout << "  输出尺寸: " << output.cols << "x" << output.rows << std::endl;

    // [CPU] 确保输出目录存在并保存
    fs::create_directories(fs::path(output_path).parent_path());
    cv::imwrite(output_path, output);
    std::cout << "  保存到: " << output_path << std::endl;
}

// ==================== 批量处理 ====================

void Inference::process_folder(const std::string& input_dir, const std::string& output_dir) {
    if (device_type_ == DeviceType::Hybrid) {
        std::cout << "\n=== GPU+CPU混合流水线批量处理 ===" << std::endl;
        process_folder_pipeline(input_dir, output_dir);
    } else {
        process_folder_sequential(input_dir, output_dir);
    }
}

// ==================== 顺序处理 (单设备) ====================

void Inference::process_folder_sequential(const std::string& input_dir, const std::string& output_dir) {
    fs::create_directories(output_dir);

    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};

    int count = 0;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) {
            continue;
        }

        std::string output_path = output_dir + "/" +
            entry.path().stem().string() + "_sr" + ext;

        try {
            process_file(entry.path().string(), output_path);
            count++;
        } catch (const std::exception& e) {
            std::cerr << "Processing failed: " << entry.path() << " - " << e.what() << std::endl;
        }
    }

    std::cout << "批量处理完成! 共处理 " << count << " 张图像" << std::endl;
}

// ==================== 流水线处理 (GPU+CPU混合) ====================

void Inference::process_folder_pipeline(const std::string& input_dir, const std::string& output_dir) {
    fs::create_directories(output_dir);

    // 收集所有待处理文件
    std::vector<std::pair<std::string, std::string>> file_pairs;
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) {
            continue;
        }

        std::string output_path = output_dir + "/" +
            entry.path().stem().string() + "_sr" + ext;
        file_pairs.emplace_back(entry.path().string(), output_path);
    }

    if (file_pairs.empty()) {
        std::cout << "未找到可处理的图像文件" << std::endl;
        return;
    }

    std::cout << "待处理图像: " << file_pairs.size() << " 张" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // 创建流水线队列
    const size_t queue_size = constants::DEFAULT_PIPELINE_QUEUE_SIZE;
    BoundedQueue<PipelineItem> preprocess_to_gpu(queue_size);   // CPU预处理 → GPU推理
    BoundedQueue<PipelineItem> gpu_to_postprocess(queue_size);  // GPU推理 → CPU后处理

    std::atomic<int> processed_count{0};
    std::atomic<int> error_count{0};

    // ============ 阶段1: CPU预处理线程 ============
    // 负责: 图像加载 (磁盘IO) + 解码 + 格式转换 + 张量化
    std::thread preprocess_thread([&]() {
        for (const auto& [in_path, out_path] : file_pairs) {
            PipelineItem item;
            item.input_path = in_path;
            item.output_path = out_path;

            try {
                // [CPU] 从磁盘加载并解码图像
                item.input_mat = cv::imread(in_path, cv::IMREAD_COLOR);
                if (item.input_mat.empty()) {
                    std::cerr << "[预处理] 无法加载: " << in_path << std::endl;
                    item.valid = false;
                    error_count++;
                    continue;
                }

                // [CPU] 预处理: Mat → Tensor (在CPU上完成)
                item.input_tensor = utils::mat_to_tensor(item.input_mat, true);

                // 使用 pin_memory 加速后续的 CPU→GPU 传输
                if (item.input_tensor.device().type() == torch::kCPU) {
                    item.input_tensor = item.input_tensor.pin_memory();
                }

            } catch (const std::exception& e) {
                std::cerr << "[预处理] 失败: " << in_path << " - " << e.what() << std::endl;
                item.valid = false;
                error_count++;
                continue;
            }

            preprocess_to_gpu.push(std::move(item));
        }

        // 发送结束信号
        preprocess_to_gpu.close();
    });

    // ============ 阶段2: GPU推理线程 ============
    // 负责: 张量传输到GPU + 模型前向传播 + 结果传回CPU
    std::thread gpu_thread([&]() {
        PipelineItem item;
        while (preprocess_to_gpu.pop(item)) {
            if (!item.valid) {
                gpu_to_postprocess.push(std::move(item));
                continue;
            }

            try {
                // [GPU] 将预处理后的张量传输到GPU (non_blocking允许CPU继续工作)
                auto gpu_tensor = item.input_tensor.to(gpu_device_, /*non_blocking=*/true);

                // [GPU] 模型前向推理
                torch::Tensor output;
                {
                    torch::NoGradGuard no_grad;
                    output = generator_->forward(gpu_tensor);
                    output = output.clamp(0, 1);
                }

                // [GPU→CPU] 将结果传回CPU (后处理在CPU上完成)
                item.output_tensor = output.to(cpu_device_);

                // 释放GPU端输入张量
                item.input_tensor = torch::Tensor();

            } catch (const std::exception& e) {
                std::cerr << "[GPU推理] 失败: " << item.input_path << " - " << e.what() << std::endl;
                item.valid = false;
                error_count++;
            }

            gpu_to_postprocess.push(std::move(item));
        }

        // 发送结束信号
        gpu_to_postprocess.close();
    });

    // ============ 阶段3: CPU后处理线程 ============
    // 负责: 张量→图像转换 + 图像编码 + 磁盘写入
    std::thread postprocess_thread([&]() {
        PipelineItem item;
        while (gpu_to_postprocess.pop(item)) {
            if (!item.valid) continue;

            try {
                // [CPU] 后处理: Tensor → Mat
                item.output_mat = utils::tensor_to_mat(item.output_tensor);

                // [CPU] 编码并保存到磁盘
                fs::create_directories(fs::path(item.output_path).parent_path());
                cv::imwrite(item.output_path, item.output_mat);

                int count = ++processed_count;
                std::cout << "[" << count << "/" << file_pairs.size() << "] "
                          << fs::path(item.input_path).filename().string()
                          << " (" << item.input_mat.cols << "x" << item.input_mat.rows
                          << " → " << item.output_mat.cols << "x" << item.output_mat.rows << ")"
                          << std::endl;

            } catch (const std::exception& e) {
                std::cerr << "[后处理] 失败: " << item.output_path << " - " << e.what() << std::endl;
                error_count++;
            }
        }
    });

    // 等待所有阶段完成
    preprocess_thread.join();
    gpu_thread.join();
    postprocess_thread.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n=== 混合流水线处理完成 ===" << std::endl;
    std::cout << "  成功: " << processed_count.load() << " 张" << std::endl;
    std::cout << "  失败: " << error_count.load() << " 张" << std::endl;
    std::cout << "  耗时: " << duration.count() / 1000.0 << " 秒" << std::endl;
    if (processed_count.load() > 0) {
        std::cout << "  平均: " << duration.count() / processed_count.load() << " ms/张" << std::endl;
    }
}

}  // namespace facesr
