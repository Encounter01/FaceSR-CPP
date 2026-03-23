#pragma once

// ============================================================================
// trainer.h — GAN训练器
//
// 管理完整的训练流程:
// 1. 构建生成器(G)和判别器(D)
// 2. 构建损失函数和优化器
// 3. 渐进式训练:
//    - 阶段1 (epoch 0-49): 仅像素损失 → 学习基本映射
//    - 阶段2 (epoch 50-149): +感知损失 → 学习纹理结构
//    - 阶段3 (epoch 150-299): +GAN损失 → 生成真实细节
// 4. 定期验证、保存检查点
//
// 每个epoch的训练步骤:
// 1. 训练判别器D: 让D能区分真实HR和生成SR
// 2. 训练生成器G: 让G能骗过D，同时保持像素/感知精度
// ============================================================================

#include <torch/torch.h>
#include <string>
#include <memory>
#include <atomic>
#include "common/config.h"
#include "common/config_parser.h"
#include "models/generator.h"
#include "models/discriminator.h"
#include "models/losses.h"
#include "utils/dataset.h"

namespace facesr {

// ============================================================================
// TrainConfig — 训练配置结构体
// 包含训练所需的所有超参数，可从INI配置文件加载
// ============================================================================
struct TrainConfig {
    // --- 数据路径和尺寸 ---
    std::string train_hr_dir = "data/train/hr";  // 训练集HR图像目录
    std::string train_lr_dir = "";                // 训练集LR目录（空=在线生成）
    std::string val_hr_dir = "data/val/hr";       // 验证集HR目录
    std::string val_lr_dir = "";                  // 验证集LR目录
    int hr_size = constants::DEFAULT_HR_SIZE;      // 256
    int lr_size = constants::DEFAULT_LR_SIZE;      // 64
    int scale = constants::DEFAULT_SCALE;          // 4

    // --- 训练超参数 ---
    int batch_size = constants::DEFAULT_BATCH_SIZE;          // 12
    int num_epochs = constants::DEFAULT_NUM_EPOCHS;          // 300
    int num_workers = constants::DEFAULT_NUM_WORKERS;        // 4
    double lr_g = constants::DEFAULT_LEARNING_RATE;          // 2e-4 生成器学习率
    double lr_d = constants::DEFAULT_LEARNING_RATE;          // 2e-4 判别器学习率

    // --- 阶段训练边界 ---
    int phase1_epochs = constants::PHASE1_END_EPOCH;  // 50: 阶段1结束
    int phase2_epochs = constants::PHASE2_END_EPOCH;  // 150: 阶段2结束

    // --- 损失权重 ---
    double pixel_weight = constants::DEFAULT_PIXEL_WEIGHT;           // 1.0
    double perceptual_weight = constants::DEFAULT_PERCEPTUAL_WEIGHT; // 1.0
    double gan_weight = constants::DEFAULT_GAN_WEIGHT;               // 0.1
    double frequency_weight = constants::DEFAULT_FREQUENCY_WEIGHT;   // 0.0 (禁用)
    double gradient_weight = constants::DEFAULT_GRADIENT_WEIGHT;     // 0.0 (禁用)
    double r1_weight = constants::DEFAULT_R1_WEIGHT;                 // 0.0 (禁用)

    // --- 损失类型 ---
    PixelLossType pixel_type = PixelLossType::L1;    // 像素损失使用L1
    GANType gan_type = GANType::Vanilla;              // GAN使用原始BCE

    // --- 保存和日志配置 ---
    std::string checkpoint_dir = "checkpoints";     // 模型权重保存目录
    std::string result_dir = "results";             // 验证结果图像保存目录
    int save_interval = constants::DEFAULT_SAVE_INTERVAL;  // 每10轮保存
    int val_interval = constants::DEFAULT_VAL_INTERVAL;    // 每5轮验证
    int log_interval = constants::DEFAULT_LOG_INTERVAL;    // 每100步打印

    // --- 网络配置 ---
    std::string vgg_weights_path = "";  // VGG19预训练权重路径（感知损失需要）
    DeviceType device_type = DeviceType::Auto;  // 计算设备
    bool use_attention = constants::DEFAULT_USE_ATTENTION;            // CBAM注意力
    bool use_spectral_norm = constants::DEFAULT_USE_SPECTRAL_NORM;   // 判别器谱归一化

    // ========================================================================
    // loadFromFile — 从INI配置文件加载所有训练参数
    // 使用ConfigParser按 "section.key" 格式读取
    // ========================================================================
    bool loadFromFile(const std::string& filepath) {
        ConfigParser parser;
        if (!parser.loadFromFile(filepath)) return false;

        // [data] 节
        train_hr_dir = parser.getString("data.train_hr_dir", train_hr_dir);
        train_lr_dir = parser.getString("data.train_lr_dir", train_lr_dir);
        val_hr_dir = parser.getString("data.val_hr_dir", val_hr_dir);
        val_lr_dir = parser.getString("data.val_lr_dir", val_lr_dir);
        hr_size = parser.getInt("data.hr_size", hr_size);
        lr_size = parser.getInt("data.lr_size", lr_size);
        scale = parser.getInt("data.scale", scale);

        // [training] 节
        batch_size = parser.getInt("training.batch_size", batch_size);
        num_epochs = parser.getInt("training.num_epochs", num_epochs);
        num_workers = parser.getInt("training.num_workers", num_workers);
        lr_g = parser.getDouble("training.lr_g", lr_g);
        lr_d = parser.getDouble("training.lr_d", lr_d);
        phase1_epochs = parser.getInt("training.phase1_epochs", phase1_epochs);
        phase2_epochs = parser.getInt("training.phase2_epochs", phase2_epochs);

        // [loss] 节
        pixel_weight = parser.getDouble("loss.pixel_weight", pixel_weight);
        perceptual_weight = parser.getDouble("loss.perceptual_weight", perceptual_weight);
        gan_weight = parser.getDouble("loss.gan_weight", gan_weight);
        frequency_weight = parser.getDouble("loss.frequency_weight", frequency_weight);
        gradient_weight = parser.getDouble("loss.gradient_weight", gradient_weight);
        r1_weight = parser.getDouble("loss.r1_weight", r1_weight);
        pixel_type = stringToPixelLossType(parser.getString("loss.pixel_type", "l1"));
        gan_type = stringToGANType(parser.getString("loss.gan_type", "vanilla"));

        // [output] 节
        checkpoint_dir = parser.getString("output.checkpoint_dir", checkpoint_dir);
        result_dir = parser.getString("output.result_dir", result_dir);
        save_interval = parser.getInt("output.save_interval", save_interval);
        val_interval = parser.getInt("output.val_interval", val_interval);
        log_interval = parser.getInt("output.log_interval", log_interval);

        // [model] 节
        vgg_weights_path = parser.getString("model.vgg_weights_path", vgg_weights_path);
        use_attention = parser.getBool("model.use_attention", use_attention);
        use_spectral_norm = parser.getBool("model.use_spectral_norm", use_spectral_norm);

        return true;
    }

    // ========================================================================
    // saveToFile — 将当前配置保存到INI文件
    // ========================================================================
    bool saveToFile(const std::string& filepath) const {
        ConfigParser parser;

        // [data] 节
        parser.set("data.train_hr_dir", train_hr_dir);
        parser.set("data.train_lr_dir", train_lr_dir);
        parser.set("data.val_hr_dir", val_hr_dir);
        parser.set("data.val_lr_dir", val_lr_dir);
        parser.set("data.hr_size", hr_size);
        parser.set("data.lr_size", lr_size);
        parser.set("data.scale", scale);

        // [training] 节
        parser.set("training.batch_size", batch_size);
        parser.set("training.num_epochs", num_epochs);
        parser.set("training.num_workers", num_workers);
        parser.set("training.lr_g", lr_g);
        parser.set("training.lr_d", lr_d);
        parser.set("training.phase1_epochs", phase1_epochs);
        parser.set("training.phase2_epochs", phase2_epochs);

        // [loss] 节
        parser.set("loss.pixel_weight", pixel_weight);
        parser.set("loss.perceptual_weight", perceptual_weight);
        parser.set("loss.gan_weight", gan_weight);
        parser.set("loss.frequency_weight", frequency_weight);
        parser.set("loss.gradient_weight", gradient_weight);
        parser.set("loss.r1_weight", r1_weight);
        parser.set("loss.pixel_type", pixelLossTypeToString(pixel_type));
        parser.set("loss.gan_type", ganTypeToString(gan_type));

        // [output] 节
        parser.set("output.checkpoint_dir", checkpoint_dir);
        parser.set("output.result_dir", result_dir);
        parser.set("output.save_interval", save_interval);
        parser.set("output.val_interval", val_interval);
        parser.set("output.log_interval", log_interval);

        return parser.saveToFile(filepath);
    }
};

// ============================================================================
// Trainer — 训练器类
// ============================================================================
class Trainer {
public:
    // 构造函数: 接收训练配置，初始化所有组件
    explicit Trainer(const TrainConfig& config);

    // ========================================================================
    // train — 完整训练流程
    // resume_path: 可选的检查点路径（文件或目录），从指定位置恢复训练
    // ========================================================================
    void train(const std::string& resume_path = "");

    // ========================================================================
    // train_epoch — 训练单个epoch
    // 返回: 该epoch的平均总损失
    // ========================================================================
    double train_epoch(int epoch);

    // ========================================================================
    // validate — 在验证集上评估模型
    // 返回: 平均PSNR值
    // ========================================================================
    double validate(int epoch);

    // 保存模型检查点（生成器和判别器的权重）
    void save_checkpoint(int epoch, const std::string& suffix = "");

    // 加载模型检查点（支持文件路径或目录路径）
    void load_checkpoint(const std::string& path);

    // --- 优雅停止接口 ---
    // 请求停止训练（线程安全，供信号处理器调用）
    static void requestStop();
    // 检查是否收到停止请求
    static bool isStopRequested();

private:
    // 停止请求标志（原子变量，线程安全）
    static std::atomic<bool> stop_requested_;

    // 保存训练状态（epoch、best_psnr、global_step + 优化器状态）
    void save_training_state(int epoch, double best_psnr);
    // 加载训练状态，返回恢复的epoch编号，通过引用返回best_psnr
    int load_training_state(double& best_psnr);
    // 根据配置初始化计算设备
    torch::Device initDevice() const;

    // 根据epoch判断训练阶段（返回是否使用感知损失和GAN损失）
    std::tuple<bool, bool> get_training_phase(int epoch) const;

    TrainConfig config_;  // 训练配置

    torch::Device device_;  // 计算设备（CPU或CUDA）

    // --- 网络模型 ---
    RRDBNet generator_{nullptr};               // RRDB生成器（约16.7M参数）
    VGGStyleDiscriminator discriminator_{nullptr};  // VGG判别器（约8.3M参数）

    // --- 损失函数 ---
    CombinedLoss loss_fn_;  // 组合损失（像素+感知+GAN）

    // --- 优化器 ---
    // Adam优化器: 自适应学习率，betas=(0.9, 0.99)
    std::unique_ptr<torch::optim::Adam> optimizer_g_;  // 生成器优化器
    std::unique_ptr<torch::optim::Adam> optimizer_d_;  // 判别器优化器

    // --- 数据加载器 ---
    // 复杂类型说明:
    // - FaceSRDataset: 我们的数据集类
    // - Stack<Example<>>: 将多个样本堆叠为batch的Transform
    // - RandomSampler: 随机采样器
    using DataLoaderType = torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            FaceSRDataset,
            torch::data::transforms::Stack<torch::data::Example<>>>,
        torch::data::samplers::RandomSampler>;

    std::unique_ptr<DataLoaderType> train_loader_;  // 训练数据加载器

    int global_step_ = 0;  // 全局训练步数（跨epoch累计）
};

}  // namespace facesr
