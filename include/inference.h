#pragma once
/**
 * @file inference.h
 * @brief 推理器类 (支持GPU+CPU混合模式)
 *
 * 支持三种运行模式:
 * - CPU模式: 所有计算在CPU上执行
 * - GPU模式: 所有计算在GPU上执行
 * - Hybrid模式: GPU执行模型推理, CPU并行处理图像IO和预处理/后处理
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "models/generator.h"
#include "common/config.h"
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>

namespace facesr {

/**
 * @brief 线程安全的有界队列, 用于流水线各阶段间的数据传递
 */
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || closed_; });
        if (closed_) return;
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::queue<T> queue_;
    size_t max_size_;
    std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    bool closed_ = false;
};

/**
 * @brief 流水线任务项
 */
struct PipelineItem {
    std::string input_path;
    std::string output_path;
    cv::Mat input_mat;           // CPU: 原始图像
    torch::Tensor input_tensor;  // CPU→GPU: 预处理后的张量
    torch::Tensor output_tensor; // GPU→CPU: 推理结果张量
    cv::Mat output_mat;          // CPU: 最终输出图像
    bool valid = true;
};

/**
 * @brief 推理器类 (支持GPU+CPU混合流水线)
 */
class Inference {
public:
    /**
     * @param model_path 模型文件路径
     * @param scale 放大倍数
     * @param use_gpu 是否使用GPU
     * @param use_attention 是否启用CBAM注意力模块
     */
    Inference(const std::string& model_path, int scale = 4,
              bool use_gpu = true, bool use_attention = false);

    /**
     * @param model_path 模型文件路径
     * @param scale 放大倍数
     * @param device_type 设备类型 (CPU/CUDA/Auto/Hybrid)
     */
    Inference(const std::string& model_path, int scale, DeviceType device_type);

    /**
     * @brief 处理单张图像
     * @param input 输入图像 (OpenCV Mat, BGR)
     * @return 输出图像 (OpenCV Mat, BGR)
     */
    cv::Mat process(const cv::Mat& input);

    /**
     * @brief 处理图像文件
     */
    bool process_file(const std::string& input_path, const std::string& output_path);

    /**
     * @brief 批量处理文件夹
     */
    int process_folder(const std::string& input_dir, const std::string& output_dir);

    bool is_loaded() const { return model_loaded_; }
    bool isModelLoaded() const { return model_loaded_; }

    /**
     * @brief 获取放大倍数
     */
    int get_scale() const { return scale_; }

    /**
     * @brief 获取当前设备类型
     */
    DeviceType get_device_type() const { return device_type_; }

    /**
     * @brief 检查是否处于混合模式
     */
    bool is_hybrid() const { return device_type_ == DeviceType::Hybrid; }

private:
    /**
     * @brief 初始化模型
     */
    void init_model(const std::string& model_path);

    /**
     * @brief 流水线批量处理 (GPU+CPU混合模式)
     *
     * 三阶段流水线:
     * 阶段1 [CPU线程池]: 图像加载 + 预处理 (mat_to_tensor)
     * 阶段2 [GPU线程]:   模型前向推理
     * 阶段3 [CPU线程池]: 后处理 (tensor_to_mat) + 图像保存
     */
    void process_folder_pipeline(const std::string& input_dir, const std::string& output_dir);

    /**
     * @brief 顺序批量处理 (CPU或GPU单设备模式)
     */
    void process_folder_sequential(const std::string& input_dir, const std::string& output_dir);

    RRDBNet generator_{nullptr};
    torch::jit::Module jit_module_;
    bool use_jit_ = false;
    torch::Device device_{torch::kCPU};
    torch::Device gpu_device_{torch::kCPU};
    torch::Device cpu_device_{torch::kCPU};
    int scale_;
    bool model_loaded_ = false;
    DeviceType device_type_;
};

}  // namespace facesr
