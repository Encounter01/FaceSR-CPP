#pragma once
/**
 * @file config.h
 * @brief 全局配置、常量和枚举定义
 *
 * 集中管理项目中的常量和类型定义，避免魔法数字
 *
 * 阅读提示：
 * 1. 先看 constants 中的尺寸、损失权重和 RRDB 参数，它们决定论文中 64x64 -> 256x256 的 4 倍超分主线。
 * 2. 再看 PixelLossType、GANType、TrainingPhase 等枚举，它们和配置文件中的字符串互相转换。
 * 3. 部分常量是后续扩展预留项，例如频域损失、梯度损失、学习率调度器和 Hybrid 流水线；当前核心训练路径主要使用像素、感知和 GAN 损失。
 */

#include <string>
#include <cstdint>

namespace facesr {

// ==================== 图像处理常量 ====================
namespace constants {
    // 图像尺寸默认值
    constexpr int DEFAULT_HR_SIZE = 256;
    constexpr int DEFAULT_LR_SIZE = 64;
    constexpr int DEFAULT_SCALE = 4;

    // 归一化参数
    constexpr double NORMALIZE_SCALE = 1.0 / 255.0;
    constexpr double DENORMALIZE_SCALE = 255.0;

    // ImageNet标准化参数
    constexpr double IMAGENET_MEAN_R = 0.485;
    constexpr double IMAGENET_MEAN_G = 0.456;
    constexpr double IMAGENET_MEAN_B = 0.406;
    constexpr double IMAGENET_STD_R = 0.229;
    constexpr double IMAGENET_STD_G = 0.224;
    constexpr double IMAGENET_STD_B = 0.225;

    // 数据增强概率
    constexpr double AUGMENT_FLIP_PROB = 0.5;
    constexpr double AUGMENT_ROTATE_PROB = 0.5;

    // 训练默认参数
    constexpr int DEFAULT_BATCH_SIZE = 12;
    constexpr int DEFAULT_NUM_WORKERS = 4;
    constexpr int DEFAULT_NUM_EPOCHS = 300;
    constexpr double DEFAULT_LEARNING_RATE = 2e-4;

    // 训练阶段 epoch 边界。
    // 论文中的三阶段训练由这两个边界控制：
    // 0~49 轮只学习像素级 LR->HR 映射，50~149 轮加入 VGG 感知约束，150 轮后进入完整 GAN 训练。
    constexpr int PHASE1_END_EPOCH = 50;    // 像素损失阶段结束
    constexpr int PHASE2_END_EPOCH = 150;   // 感知损失阶段结束

    // 损失权重默认值。
    // 频域损失和梯度损失目前只是配置层预留，CombinedLoss 当前不会实际计算这两项。
    constexpr double DEFAULT_PIXEL_WEIGHT = 1.0;
    constexpr double DEFAULT_PERCEPTUAL_WEIGHT = 1.0;
    constexpr double DEFAULT_GAN_WEIGHT = 0.1;
    constexpr double DEFAULT_FREQUENCY_WEIGHT = 0.0;
    constexpr double DEFAULT_GRADIENT_WEIGHT = 0.0;
    constexpr double DEFAULT_R1_WEIGHT = 0.0;

    // GAN标签值
    constexpr double REAL_LABEL_VALUE = 1.0;
    constexpr double FAKE_LABEL_VALUE = 0.0;

    // VGG特征提取层
    constexpr int DEFAULT_VGG_FEATURE_LAYER = 35;

    // RRDB 网络参数。
    // 默认 23 个 RRDB 块、64 个主干通道、32 个增长通道，对应论文中采用的 ESRGAN 风格生成器主干。
    constexpr int RRDB_NUM_FEATURES = 64;
    constexpr int RRDB_NUM_BLOCKS = 23;
    constexpr int RRDB_GROWTH_CHANNELS = 32;

    // 学习率调度器默认参数
    constexpr double DEFAULT_LR_MIN = 1e-7;            // 最小学习率
    constexpr int DEFAULT_LR_WARMUP_EPOCHS = 0;        // 预热epoch数
    constexpr double DEFAULT_LR_DECAY_GAMMA = 0.5;     // StepLR衰减因子

    // 日志间隔
    constexpr int DEFAULT_LOG_INTERVAL = 100;
    constexpr int DEFAULT_VAL_INTERVAL = 2;
    constexpr int DEFAULT_SAVE_INTERVAL = 10;

    // GPU+CPU混合模式参数
    constexpr int DEFAULT_PIPELINE_QUEUE_SIZE = 8;
    constexpr int DEFAULT_CPU_PREPROCESS_THREADS = 2;
    constexpr int DEFAULT_CPU_POSTPROCESS_THREADS = 2;

    // 网络配置默认值
    constexpr bool DEFAULT_USE_ATTENTION = false;
    constexpr bool DEFAULT_USE_SPECTRAL_NORM = false;

    // SSIM计算参数
    constexpr double SSIM_K1 = 0.01;
    constexpr double SSIM_K2 = 0.03;
    constexpr double SSIM_L = 1.0;  // 像素值范围 [0, 1]
    constexpr int SSIM_WINDOW_SIZE = 11;
    constexpr double SSIM_SIGMA = 1.5;

    // 支持的图像格式
    inline const char* const SUPPORTED_EXTENSIONS[] = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff"
    };
    constexpr size_t NUM_SUPPORTED_EXTENSIONS = 5;
}

// ==================== 枚举类型定义 ====================

/**
 * @brief 像素损失类型
 */
enum class PixelLossType {
    L1,     // 绝对值损失
    L2      // 均方误差损失
};

/**
 * @brief GAN类型
 */
enum class GANType {
    Vanilla,    // 原始GAN (BCE损失)
    LSGAN,      // 最小二乘GAN
    WGAN,       // Wasserstein GAN
    WGAN_GP,    // 带梯度惩罚的WGAN；当前 GANLoss 中按 WGAN 形式计算，GP 项未在这里实现
    Hinge       // Hinge损失GAN
};

/**
 * @brief 插值方法
 */
enum class InterpolationMode {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos
};

/**
 * @brief 训练阶段
 */
enum class TrainingPhase {
    PixelOnly,      // 阶段1: 仅像素损失
    WithPerceptual, // 阶段2: 像素+感知损失
    Full            // 阶段3: 完整GAN训练
};

/**
 * @brief 设备类型
 */
enum class DeviceType {
    CPU,
    CUDA,
    Auto,   // 自动选择 (优先GPU)
    Hybrid  // GPU+CPU协同概念；当前命令行推理主要按 CUDA/CPU 顺序执行，完整流水线是预留扩展
};

/**
 * @brief 学习率调度器类型
 */
enum class LRSchedulerType {
    None,           // 不使用调度器 (固定学习率)
    CosineAnnealing,// 余弦退火
    Step,           // 固定步长衰减
    MultiStep       // 多步衰减
};

// ==================== 枚举转换工具函数 ====================

/**
 * @brief 将字符串转换为PixelLossType
 */
inline PixelLossType stringToPixelLossType(const std::string& str) {
    if (str == "l1" || str == "L1") return PixelLossType::L1;
    if (str == "l2" || str == "L2" || str == "mse" || str == "MSE") return PixelLossType::L2;
    return PixelLossType::L1;  // 默认
}

/**
 * @brief 将PixelLossType转换为字符串
 */
inline std::string pixelLossTypeToString(PixelLossType type) {
    switch (type) {
        case PixelLossType::L1: return "l1";
        case PixelLossType::L2: return "l2";
        default: return "l1";
    }
}

/**
 * @brief 将字符串转换为GANType
 */
inline GANType stringToGANType(const std::string& str) {
    if (str == "vanilla" || str == "Vanilla") return GANType::Vanilla;
    if (str == "lsgan" || str == "LSGAN") return GANType::LSGAN;
    if (str == "wgan" || str == "WGAN") return GANType::WGAN;
    if (str == "wgan-gp" || str == "WGAN-GP") return GANType::WGAN_GP;
    if (str == "hinge" || str == "Hinge") return GANType::Hinge;
    return GANType::Vanilla;  // 默认
}

/**
 * @brief 将GANType转换为字符串
 */
inline std::string ganTypeToString(GANType type) {
    switch (type) {
        case GANType::Vanilla: return "vanilla";
        case GANType::LSGAN: return "lsgan";
        case GANType::WGAN: return "wgan";
        case GANType::WGAN_GP: return "wgan-gp";
        case GANType::Hinge: return "hinge";
        default: return "vanilla";
    }
}

/**
 * @brief 将字符串转换为LRSchedulerType
 */
inline LRSchedulerType stringToLRSchedulerType(const std::string& str) {
    if (str == "cosine" || str == "CosineAnnealing") return LRSchedulerType::CosineAnnealing;
    if (str == "step" || str == "Step") return LRSchedulerType::Step;
    if (str == "multistep" || str == "MultiStep") return LRSchedulerType::MultiStep;
    if (str == "none" || str == "None") return LRSchedulerType::None;
    return LRSchedulerType::CosineAnnealing;  // 默认
}

/**
 * @brief 将LRSchedulerType转换为字符串
 */
inline std::string lrSchedulerTypeToString(LRSchedulerType type) {
    switch (type) {
        case LRSchedulerType::CosineAnnealing: return "cosine";
        case LRSchedulerType::Step: return "step";
        case LRSchedulerType::MultiStep: return "multistep";
        case LRSchedulerType::None: return "none";
        default: return "cosine";
    }
}

/**
 * @brief 根据epoch获取训练阶段
 */
inline TrainingPhase getTrainingPhase(int epoch) {
    if (epoch < constants::PHASE1_END_EPOCH) {
        return TrainingPhase::PixelOnly;
    } else if (epoch < constants::PHASE2_END_EPOCH) {
        return TrainingPhase::WithPerceptual;
    } else {
        return TrainingPhase::Full;
    }
}

}  // namespace facesr
