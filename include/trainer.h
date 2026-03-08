#pragma once
/**
 * @file trainer.h
 * @brief 训练器类
 *
 * 管理GAN模型的训练流程
 */

#include <torch/torch.h>
#include "models/generator.h"
#include "models/discriminator.h"
#include "models/losses.h"
#include "utils/dataset.h"
#include "utils/metrics.h"
#include "common/config.h"
#include "common/config_parser.h"
#include <string>
#include <memory>
#include <atomic>

namespace facesr {

/**
 * @brief 训练配置
 */
struct TrainConfig {
    // 数据配置
    std::string train_hr_dir = "data/train/HR";
    std::string train_lr_dir = "data/train/LR";
    std::string val_hr_dir = "data/val/HR";
    std::string val_lr_dir = "data/val/LR";
    int hr_size = constants::DEFAULT_HR_SIZE;
    int lr_size = constants::DEFAULT_LR_SIZE;
    int scale = constants::DEFAULT_SCALE;

    // 训练配置
    int batch_size = constants::DEFAULT_BATCH_SIZE;
    int num_epochs = constants::DEFAULT_NUM_EPOCHS;
    int num_workers = constants::DEFAULT_NUM_WORKERS;
    double lr_g = constants::DEFAULT_LEARNING_RATE;
    double lr_d = constants::DEFAULT_LEARNING_RATE;

    // 阶段训练
    int phase1_epochs = constants::PHASE1_END_EPOCH;    // 仅像素损失
    int phase2_epochs = constants::PHASE2_END_EPOCH;    // 像素+感知损失
    // phase3: 全部损失

    // 损失权重
    double pixel_weight = constants::DEFAULT_PIXEL_WEIGHT;
    double perceptual_weight = constants::DEFAULT_PERCEPTUAL_WEIGHT;
    double gan_weight = constants::DEFAULT_GAN_WEIGHT;

    // 学习率调度器
    LRSchedulerType lr_scheduler_type = LRSchedulerType::CosineAnnealing;
    double lr_min = constants::DEFAULT_LR_MIN;
    int lr_warmup_epochs = constants::DEFAULT_LR_WARMUP_EPOCHS;
    double lr_decay_gamma = constants::DEFAULT_LR_DECAY_GAMMA;  // Step衰减因子
    int lr_decay_step = 50;    // Step衰减间隔(epochs)

    // 损失类型
    PixelLossType pixel_loss_type = PixelLossType::L1;
    GANType gan_type = GANType::Vanilla;

    // 保存配置
    std::string checkpoint_dir = "checkpoints";
    std::string result_dir = "results";
    int save_interval = constants::DEFAULT_SAVE_INTERVAL;
    int latest_save_interval = 1;
    int val_interval = constants::DEFAULT_VAL_INTERVAL;
    int log_interval = constants::DEFAULT_LOG_INTERVAL;
    bool auto_resume = true;

    // VGG权重路径
    std::string vgg_weights_path = "";

    // 设备配置 (默认使用Hybrid混合模式, GPU推理+CPU数据加载)
    DeviceType device_type = DeviceType::Hybrid;

    // 是否使用pin_memory加速CPU→GPU数据传输
    bool pin_memory = true;

    /**
     * @brief 从配置文件加载
     */
    bool loadFromFile(const std::string& filepath) {
        ConfigParser parser;
        if (!parser.loadFromFile(filepath)) {
            return false;
        }

        // 数据配置
        train_hr_dir = parser.getString("data.train_hr_dir", train_hr_dir);
        train_lr_dir = parser.getString("data.train_lr_dir", train_lr_dir);
        val_hr_dir = parser.getString("data.val_hr_dir", val_hr_dir);
        val_lr_dir = parser.getString("data.val_lr_dir", val_lr_dir);
        hr_size = parser.getInt("data.hr_size", hr_size);
        lr_size = parser.getInt("data.lr_size", lr_size);
        scale = parser.getInt("data.scale", scale);

        // 训练配置
        batch_size = parser.getInt("training.batch_size", batch_size);
        num_epochs = parser.getInt("training.num_epochs", num_epochs);
        num_workers = parser.getInt("training.num_workers", num_workers);
        lr_g = parser.getDouble("training.lr_g", lr_g);
        lr_d = parser.getDouble("training.lr_d", lr_d);

        // 阶段配置
        phase1_epochs = parser.getInt("training.phase1_epochs", phase1_epochs);
        phase2_epochs = parser.getInt("training.phase2_epochs", phase2_epochs);

        // 学习率调度器
        lr_scheduler_type = stringToLRSchedulerType(parser.getString("training.lr_scheduler", "cosine"));
        lr_min = parser.getDouble("training.lr_min", lr_min);
        lr_warmup_epochs = parser.getInt("training.lr_warmup_epochs", lr_warmup_epochs);
        lr_decay_gamma = parser.getDouble("training.lr_decay_gamma", lr_decay_gamma);
        lr_decay_step = parser.getInt("training.lr_decay_step", lr_decay_step);

        // 损失配置
        pixel_weight = parser.getDouble("loss.pixel_weight", pixel_weight);
        perceptual_weight = parser.getDouble("loss.perceptual_weight", perceptual_weight);
        gan_weight = parser.getDouble("loss.gan_weight", gan_weight);
        pixel_loss_type = stringToPixelLossType(parser.getString("loss.pixel_type", "l1"));
        gan_type = stringToGANType(parser.getString("loss.gan_type", "vanilla"));

        // 保存配置
        checkpoint_dir = parser.getString("output.checkpoint_dir", checkpoint_dir);
        result_dir = parser.getString("output.result_dir", result_dir);
        save_interval = parser.getInt("output.save_interval", save_interval);
        latest_save_interval = parser.getInt("output.latest_save_interval", latest_save_interval);
        val_interval = parser.getInt("output.val_interval", val_interval);
        log_interval = parser.getInt("output.log_interval", log_interval);
        auto_resume = parser.getBool("output.auto_resume", auto_resume);

        // 其他配置
        vgg_weights_path = parser.getString("model.vgg_weights", vgg_weights_path);

        return true;
    }

    /**
     * @brief 保存到配置文件
     */
    bool saveToFile(const std::string& filepath) const {
        ConfigParser parser;

        // 数据配置
        parser.set("data.train_hr_dir", train_hr_dir);
        parser.set("data.train_lr_dir", train_lr_dir);
        parser.set("data.val_hr_dir", val_hr_dir);
        parser.set("data.val_lr_dir", val_lr_dir);
        parser.set("data.hr_size", hr_size);
        parser.set("data.lr_size", lr_size);
        parser.set("data.scale", scale);

        // 训练配置
        parser.set("training.batch_size", batch_size);
        parser.set("training.num_epochs", num_epochs);
        parser.set("training.num_workers", num_workers);
        parser.set("training.lr_g", lr_g);
        parser.set("training.lr_d", lr_d);
        parser.set("training.phase1_epochs", phase1_epochs);
        parser.set("training.phase2_epochs", phase2_epochs);
        parser.set("training.lr_scheduler", lrSchedulerTypeToString(lr_scheduler_type));
        parser.set("training.lr_min", lr_min);
        parser.set("training.lr_warmup_epochs", lr_warmup_epochs);
        parser.set("training.lr_decay_gamma", lr_decay_gamma);
        parser.set("training.lr_decay_step", lr_decay_step);

        // 损失配置
        parser.set("loss.pixel_weight", pixel_weight);
        parser.set("loss.perceptual_weight", perceptual_weight);
        parser.set("loss.gan_weight", gan_weight);
        parser.set("loss.pixel_type", pixelLossTypeToString(pixel_loss_type));
        parser.set("loss.gan_type", ganTypeToString(gan_type));

        // 保存配置
        parser.set("output.checkpoint_dir", checkpoint_dir);
        parser.set("output.result_dir", result_dir);
        parser.set("output.save_interval", save_interval);
        parser.set("output.latest_save_interval", latest_save_interval);
        parser.set("output.val_interval", val_interval);
        parser.set("output.log_interval", log_interval);
        parser.set("output.auto_resume", auto_resume);

        // 其他配置
        parser.set("model.vgg_weights", vgg_weights_path);

        return parser.saveToFile(filepath);
    }
};


/**
 * @brief 训练器类
 */
class Trainer {
public:
    /**
     * @param config 训练配置
     */
    explicit Trainer(const TrainConfig& config);

    /**
     * @brief 开始训练
     * @param resume_path 恢复训练的检查点路径 (可选)
     */
    void train(const std::string& resume_path = "");

    /**
     * @brief 训练单个epoch
     * @param epoch 当前epoch
     * @return 平均损失
     */
    double train_epoch(int epoch);

    /**
     * @brief 验证
     * @param epoch 当前epoch
     * @return 平均PSNR
     */
    double validate(int epoch);

    /**
     * @brief 保存检查点
     */
    void save_checkpoint(int epoch, bool is_best = false, bool save_epoch_snapshot = true);

    /**
     * @brief 加载检查点
     */
    void load_checkpoint(const std::string& path);

    /**
     * @brief 获取当前配置
     */
    const TrainConfig& getConfig() const { return config_; }

    /**
     * @brief 获取当前设备
     */
    torch::Device getDevice() const { return device_; }

    /**
     * @brief 请求停止训练（线程安全，供信号处理器调用）
     */
    static void requestStop();

    /**
     * @brief 检查是否收到停止请求
     */
    static bool isStopRequested();

private:
    // 停止请求标志（原子变量，线程安全）
    static std::atomic<bool> stop_requested_;

    /**
     * @brief 保存训练状态（epoch、best_psnr、global_step + 优化器状态）
     */
    void save_training_state(int epoch, double best_psnr);

    /**
     * @brief 加载训练状态，返回恢复的epoch编号，通过引用返回best_psnr
     */
    int load_training_state(double& best_psnr);
    bool has_latest_checkpoint(const std::string& checkpoint_dir) const;
    /**
     * @brief 获取当前训练阶段
     */
    std::tuple<bool, bool> get_training_phase(int epoch);

    /**
     * @brief 更新学习率 (根据调度器策略)
     */
    void update_learning_rate(int epoch);

    /**
     * @brief 获取当前学习率
     */
    double get_current_lr() const;

    /**
     * @brief 初始化设备
     */
    torch::Device initDevice();

    TrainConfig config_;
    torch::Device device_;

    // 模型
    models::RRDBNet generator_{nullptr};
    models::VGGStyleDiscriminator discriminator_{nullptr};

    // 损失函数
    std::shared_ptr<models::CombinedLoss> loss_fn_;

    // 优化器
    std::unique_ptr<torch::optim::Adam> optimizer_g_;
    std::unique_ptr<torch::optim::Adam> optimizer_d_;

    // 数据加载器
    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            utils::FaceSRDataset,
            torch::data::transforms::Stack<torch::data::Example<>>
        >,
        torch::data::samplers::RandomSampler
    >> train_loader_;

    int global_step_ = 0;
    bool last_epoch_interrupted_ = false;
};

}  // namespace facesr
