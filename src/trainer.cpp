// ============================================================================
// trainer.cpp — GAN训练器实现
//
// GAN训练的核心循环:
// 每个batch:
//   1. 训练判别器D:
//      - D(real_HR) → 应接近1 → loss_real = BCE(D(HR), 1)
//      - D(G(LR))  → 应接近0 → loss_fake = BCE(D(SR), 0)
//      - D_loss = (loss_real + loss_fake) / 2
//   2. 训练生成器G:
//      - SR = G(LR)
//      - pixel_loss = L1(SR, HR)
//      - perceptual_loss = L1(VGG(SR), VGG(HR))  [可选]
//      - gan_loss = BCE(D(SR), 1)                  [可选]
//      - G_loss = pixel_w * pixel + perc_w * perceptual + gan_w * gan
// ============================================================================

#include "trainer.h"
#include "common/logger.h"
#include "utils/image_utils.h"
#include "utils/metrics.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace facesr {

// 初始化静态停止标志
std::atomic<bool> Trainer::stop_requested_{false};

void Trainer::requestStop() {
    stop_requested_.store(true);
}

bool Trainer::isStopRequested() {
    return stop_requested_.load();
}

// ============================================================================
// initDevice — 初始化计算设备
// ============================================================================
torch::Device Trainer::initDevice() const {
    switch (config_.device_type) {
        case DeviceType::CPU:
            return torch::Device(torch::kCPU);
        case DeviceType::CUDA:
            // 检查CUDA是否可用，不可用则回退到CPU
            if (torch::cuda::is_available()) {
                return torch::Device(torch::kCUDA);
            }
            LOG_WARN("CUDA requested but not available, falling back to CPU");
            return torch::Device(torch::kCPU);
        case DeviceType::Auto:
        default:
            // 自动选择: 有CUDA用CUDA，否则用CPU
            return torch::cuda::is_available()
                ? torch::Device(torch::kCUDA)
                : torch::Device(torch::kCPU);
    }
}

// ============================================================================
// Trainer构造函数 — 初始化所有训练组件
// ============================================================================
Trainer::Trainer(const TrainConfig& config)
    : config_(config)
    , device_(initDevice())
    , loss_fn_(config.pixel_weight, config.perceptual_weight, config.gan_weight,
               config.pixel_type, config.gan_type) {

    LOG_INFO("Using device: ", device_.is_cuda() ? "CUDA" : "CPU");

    // 创建输出目录（如果不存在）
    fs::create_directories(config_.checkpoint_dir);
    fs::create_directories(config_.result_dir);

    // 构建生成器: RRDB网络
    // 参数: 输入3通道, 输出3通道, 64个特征通道, 23个RRDB块, 增长通道32, 4倍放大
    generator_ = RRDBNet(3, 3, constants::RRDB_NUM_FEATURES,
                         constants::RRDB_NUM_BLOCKS,
                         constants::RRDB_GROWTH_CHANNELS,
                         config_.scale,
                         config_.use_attention);
    generator_->to(device_);  // 将模型参数移到目标设备（GPU/CPU）

    // 构建判别器: VGG风格判别器（可选谱归一化）
    discriminator_ = VGGStyleDiscriminator(3, 64, config_.hr_size,
                                            config_.use_spectral_norm);
    discriminator_->to(device_);

    // 打印参数量（用于确认模型规模）
    LOG_INFO("Generator parameters: ", generator_->get_num_parameters() / 1e6, "M");
    LOG_INFO("Discriminator parameters: ", discriminator_->get_num_parameters() / 1e6, "M");

    // 构建组合损失函数
    // 损失函数内部的VGG特征提取器也需要移到对应设备
    // （通过Module注册机制自动处理）

    // 如果提供了VGG预训练权重路径，加载权重
    if (!config_.vgg_weights_path.empty()) {
        LOG_INFO("Loading VGG weights from: ", config_.vgg_weights_path);
        loss_fn_.load_vgg_weights(config_.vgg_weights_path);
    }
    loss_fn_.to(device_);

    // 构建Adam优化器
    // Adam: 自适应学习率优化器，结合了动量(Momentum)和RMSProp的优点
    // betas = (0.9, 0.99): 一阶和二阶矩的指数衰减率
    // 0.9: 动量衰减，控制梯度方向的平滑
    // 0.99: 自适应学习率衰减，控制学习率的自适应调整
    optimizer_g_ = std::make_unique<torch::optim::Adam>(
        generator_->parameters(),
        torch::optim::AdamOptions(config_.lr_g).betas({0.9, 0.99}));

    optimizer_d_ = std::make_unique<torch::optim::Adam>(
        discriminator_->parameters(),
        torch::optim::AdamOptions(config_.lr_d).betas({0.9, 0.99}));

    // 构建数据加载器
    // 1. 创建FaceSRDataset（启用数据增强）
    auto dataset = utils::FaceSRDataset(config_.train_hr_dir, config_.train_lr_dir,
                                  config_.hr_size, config_.lr_size, true)
        // 2. .map(Stack<>): 将多个 Example<> 堆叠为 batch
        // Stack会把多个 [3,64,64] 的LR张量堆叠为 [B,3,64,64]
        .map(torch::data::transforms::Stack<torch::data::Example<>>());

    // 3. make_data_loader: 创建多线程数据加载器
    // RandomSampler: 每个epoch随机打乱数据顺序
    train_loader_ = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(config_.batch_size)
            .workers(config_.num_workers)
            .enforce_ordering(false));  // 不强制顺序，提高多线程效率
}

// ============================================================================
// get_training_phase — 根据epoch判断训练阶段
//
// 返回 (use_perceptual, use_gan) 元组:
// - PixelOnly: (false, false) — 仅像素损失
// - WithPerceptual: (true, false) — 像素 + 感知损失
// - Full: (true, true) — 像素 + 感知 + GAN损失
// ============================================================================
std::tuple<bool, bool> Trainer::get_training_phase(int epoch) const {
    if (epoch < config_.phase1_epochs) {
        return {false, false};  // 阶段1: 仅像素损失
    } else if (epoch < config_.phase2_epochs) {
        return {true, false};   // 阶段2: 像素 + 感知
    } else {
        return {true, true};    // 阶段3: 完整GAN训练
    }
}

// ============================================================================
// train_epoch — 训练单个epoch
//
// 返回该epoch的平均总损失
// ============================================================================
double Trainer::train_epoch(int epoch) {
    generator_->train();      // 设置生成器为训练模式（启用Dropout/BN训练行为）
    discriminator_->train();  // 设置判别器为训练模式

    // 获取当前阶段的损失开关
    auto [use_perceptual, use_gan] = get_training_phase(epoch);

    AverageMeter loss_meter;  // 记录平均损失

    // 遍历每个batch
    for (auto& batch : *train_loader_) {
        // 检查是否收到停止请求
        if (isStopRequested()) {
            LOG_INFO("停止请求已接收，正在结束当前epoch...");
            break;
        }

        auto lr_data = batch.data.to(device_);    // LR图像移到设备: [B, 3, 64, 64]
        auto hr_data = batch.target.to(device_);  // HR图像移到设备: [B, 3, 256, 256]

        // ================================================================
        // 第1步: 训练判别器 D
        // 目标: 让D能正确区分真实HR图像和生成的SR图像
        // ================================================================
        // Keep pixel/perceptual-only phases from updating the discriminator.
        if (use_gan) {
            optimizer_d_->zero_grad();

            // Generate SR without generator gradients for discriminator training.
            torch::Tensor sr_for_d;
            {
                torch::NoGradGuard no_grad;
                sr_for_d = generator_->forward(lr_data);  // G(LR) = SR
            }

            auto real_images = hr_data;
            if (config_.r1_weight > 0.0) {
                real_images = real_images.detach().requires_grad_(true);
            }

            auto disc_real = discriminator_->forward(real_images);
            auto loss_real = models::GANLoss(config_.gan_type).forward(disc_real, true, true);

            // Detach SR so discriminator updates do not backprop into G.
            auto disc_fake = discriminator_->forward(sr_for_d.detach());
            auto loss_fake = models::GANLoss(config_.gan_type).forward(disc_fake, false, true);

            auto d_loss = (loss_real + loss_fake) * 0.5;

            // R1 penalty is only meaningful during adversarial training.
            if (config_.r1_weight > 0.0) {
                auto grad_outputs = torch::ones_like(disc_real);
                auto grads = torch::autograd::grad(
                    {disc_real.sum()},
                    {real_images},
                    {grad_outputs},             // grad_outputs
                    /*retain_graph=*/true,
                    /*create_graph=*/true
                );
                auto r1_penalty = grads[0].square().view({real_images.size(0), -1}).sum(1).mean();
                d_loss = d_loss + config_.r1_weight * r1_penalty;
            }

            d_loss.backward();
            optimizer_d_->step();
        }

        // ================================================================
        // 第2步: 训练生成器 G
        // 目标: 最小化组合损失（像素+感知+GAN）
        // ================================================================
        optimizer_g_->zero_grad();  // 清零生成器梯度

        // 生成超分辨率图像（这次需要梯度，因为要训练G）
        auto sr = generator_->forward(lr_data);

        // 如果使用GAN损失，让判别器评估生成的图像
        // 注意: 这里不detach，因为需要梯度流回生成器
        torch::Tensor disc_pred_fake;
        if (use_gan) {
            disc_pred_fake = discriminator_->forward(sr);
        }

        // 计算组合损失
        auto losses = loss_fn_.forward(sr, hr_data, disc_pred_fake,
                                       use_perceptual, use_gan);

        losses["total"].backward();  // 反向传播
        optimizer_g_->step();        // 更新生成器参数

        // 记录损失值
        loss_meter.update(losses["total"].item<double>(), lr_data.size(0));

        // 按间隔打印进度
        ++global_step_;
        if (global_step_ % config_.log_interval == 0) {
            LOG_INFO("Epoch [", epoch, "] Step [", global_step_, "] ",
                     "Loss: ", losses["total"].item<double>(),
                     " (pixel: ", losses["pixel"].item<double>(),
                     ", perceptual: ", losses["perceptual"].item<double>(),
                     ", gan: ", losses["gan"].item<double>(), ")");
        }
    }

    return loss_meter.avg();  // 返回该epoch的平均损失
}

// ============================================================================
// validate — 在验证集上评估模型
//
// 使用PSNR和SSIM指标，同时保存第一张验证结果图像
// ============================================================================
double Trainer::validate(int epoch) {
    generator_->eval();  // 设置为评估模式（关闭Dropout，BN使用滑动统计量）

    utils::MetricCalculator metrics;

    // 创建验证数据集（无数据增强，batch_size=1）
    auto val_dataset = utils::FaceSRDataset(config_.val_hr_dir, config_.val_lr_dir,
                                      config_.hr_size, config_.lr_size, false)
        .map(torch::data::transforms::Stack<torch::data::Example<>>());

    auto val_loader = torch::data::make_data_loader(
        std::move(val_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(1)
            .workers(1)
            .enforce_ordering(true));

    bool first_saved = false;  // 是否已保存第一张对比图

    // 遍历验证集
    for (auto& batch : *val_loader) {
        auto lr_data = batch.data.to(device_);
        auto hr_data = batch.target.to(device_);

        // NoGradGuard: 验证时不需要计算梯度，节省内存和计算
        torch::Tensor sr;
        {
            torch::NoGradGuard no_grad;
            sr = generator_->forward(lr_data);
        }

        // clamp到[0,1]（生成器输出可能略超出范围）
        sr = sr.clamp(0.0, 1.0);

        // 计算PSNR和SSIM
        metrics.update(sr, hr_data);

        // 保存第一张验证结果的对比图（LR vs SR vs HR）
        if (!first_saved) {
            auto comparison = utils::create_comparison_image(
                lr_data.cpu(), sr.cpu(), hr_data.cpu());
            std::string save_path = config_.result_dir + "/val_epoch_" +
                                     std::to_string(epoch) + ".png";
            cv::imwrite(save_path, comparison);
            first_saved = true;
        }
    }

    // 打印平均指标
    double avg_psnr = metrics.get_avg_psnr();
    double avg_ssim = metrics.get_avg_ssim();
    LOG_INFO("Validation Epoch [", epoch, "] ",
             "PSNR: ", avg_psnr, " dB, SSIM: ", avg_ssim);

    generator_->train();  // 恢复训练模式

    return avg_psnr;
}

// ============================================================================
// save_checkpoint — 保存模型检查点
//
// 保存策略:
// - epochN: 带epoch编号的版本（定期备份）
// - latest: 最新版本（方便恢复训练）
// - best: 最佳版本（验证PSNR最高时保存）
// ============================================================================
void Trainer::save_checkpoint(int epoch, const std::string& suffix) {
    std::string gen_path = config_.checkpoint_dir + "/generator";
    std::string disc_path = config_.checkpoint_dir + "/discriminator";

    if (suffix.empty()) {
        gen_path += "_epoch" + std::to_string(epoch) + ".pt";
        disc_path += "_epoch" + std::to_string(epoch) + ".pt";
    } else {
        gen_path += "_" + suffix + ".pt";
        disc_path += "_" + suffix + ".pt";
    }

    // 保存模型参数
    torch::save(generator_, gen_path);
    torch::save(discriminator_, disc_path);

    // 同时保存latest版本
    torch::save(generator_, config_.checkpoint_dir + "/generator_latest.pt");
    torch::save(discriminator_, config_.checkpoint_dir + "/discriminator_latest.pt");

    LOG_INFO("Checkpoint saved: ", gen_path);
}

// ============================================================================
// save_training_state — 保存训练状态（epoch、best_psnr、global_step + 优化器）
// ============================================================================
void Trainer::save_training_state(int epoch, double best_psnr) {
    // 保存训练状态到二进制文件
    std::string state_path = config_.checkpoint_dir + "/train_state.bin";
    std::ofstream ofs(state_path, std::ios::binary);
    if (ofs.is_open()) {
        int32_t ep = epoch;
        int32_t gs = global_step_;
        ofs.write(reinterpret_cast<const char*>(&ep), sizeof(ep));
        ofs.write(reinterpret_cast<const char*>(&best_psnr), sizeof(best_psnr));
        ofs.write(reinterpret_cast<const char*>(&gs), sizeof(gs));
        ofs.close();
        LOG_INFO("Training state saved: epoch=", epoch,
                 ", best_psnr=", best_psnr, ", global_step=", global_step_);
    } else {
        LOG_WARN("Failed to save training state to: ", state_path);
    }

    // 保存优化器状态
    try {
        std::string opt_g_path = config_.checkpoint_dir + "/optimizer_g_state.pt";
        std::string opt_d_path = config_.checkpoint_dir + "/optimizer_d_state.pt";

        torch::serialize::OutputArchive archive_g;
        optimizer_g_->save(archive_g);
        archive_g.save_to(opt_g_path);

        torch::serialize::OutputArchive archive_d;
        optimizer_d_->save(archive_d);
        archive_d.save_to(opt_d_path);

        LOG_INFO("Optimizer states saved");
    } catch (const std::exception& e) {
        LOG_WARN("Failed to save optimizer states: ", e.what());
    }
}

// ============================================================================
// load_training_state — 加载训练状态
// 返回恢复的epoch编号，通过引用返回best_psnr
// ============================================================================
int Trainer::load_training_state(double& best_psnr) {
    std::string state_path = config_.checkpoint_dir + "/train_state.bin";
    if (!fs::exists(state_path)) {
        LOG_WARN("Training state file not found: ", state_path);
        return 0;
    }

    std::ifstream ifs(state_path, std::ios::binary);
    if (!ifs.is_open()) {
        LOG_WARN("Failed to open training state file: ", state_path);
        return 0;
    }

    int32_t epoch = 0;
    int32_t gs = 0;
    ifs.read(reinterpret_cast<char*>(&epoch), sizeof(epoch));
    ifs.read(reinterpret_cast<char*>(&best_psnr), sizeof(best_psnr));
    ifs.read(reinterpret_cast<char*>(&gs), sizeof(gs));
    ifs.close();

    global_step_ = gs;
    LOG_INFO("Training state loaded: epoch=", epoch,
             ", best_psnr=", best_psnr, ", global_step=", global_step_);

    // 加载优化���状态
    try {
        std::string opt_g_path = config_.checkpoint_dir + "/optimizer_g_state.pt";
        std::string opt_d_path = config_.checkpoint_dir + "/optimizer_d_state.pt";

        if (fs::exists(opt_g_path) && fs::exists(opt_d_path)) {
            torch::serialize::InputArchive archive_g;
            archive_g.load_from(opt_g_path);
            optimizer_g_->load(archive_g);

            torch::serialize::InputArchive archive_d;
            archive_d.load_from(opt_d_path);
            optimizer_d_->load(archive_d);

            LOG_INFO("Optimizer states loaded");
        } else {
            LOG_WARN("Optimizer state files not found, using fresh optimizers");
        }
    } catch (const std::exception& e) {
        LOG_WARN("Failed to load optimizer states: ", e.what(),
                 " — using fresh optimizers");
    }

    return epoch;
}

// ============================================================================
// load_checkpoint — 加载模型检查点（支持文件路径或目录路径）
// ============================================================================
void Trainer::load_checkpoint(const std::string& path) {
    std::string gen_path, disc_path;

    if (fs::is_directory(path)) {
        // 目录路径: 加载latest版本
        gen_path = path + "/generator_latest.pt";
        disc_path = path + "/discriminator_latest.pt";
    } else {
        // 文件路径: 直接使用（向后兼容）
        gen_path = path;
        // 尝试从同目录加载判别器
        auto dir = fs::path(path).parent_path().string();
        disc_path = dir + "/discriminator_latest.pt";
    }

    LOG_INFO("Loading generator checkpoint: ", gen_path);
    torch::load(generator_, gen_path);
    generator_->to(device_);

    // 加载判别器（如果存在）
    if (fs::exists(disc_path)) {
        LOG_INFO("Loading discriminator checkpoint: ", disc_path);
        torch::load(discriminator_, disc_path);
        discriminator_->to(device_);
    } else {
        LOG_WARN("Discriminator checkpoint not found: ", disc_path);
    }
}

// ============================================================================
// train — 完整训练流程
//
// 300个epoch的渐进式训练:
// epoch 0-49: 仅像素损失（L1）— 学习低频信息
// epoch 50-149: +感知损失（VGG）— 学习中高频纹理
// epoch 150-299: +GAN损失 — 生成真实感细节
// ============================================================================
void Trainer::train(const std::string& resume_path) {
    int start_epoch = 0;
    double best_psnr = 0.0;  // 记录最佳验证PSNR

    // 如果提供了恢复路径，加载检查点和训练状态
    if (!resume_path.empty()) {
        load_checkpoint(resume_path);
        // 加载训练状态（epoch、best_psnr、global_step、优化器状态）
        start_epoch = load_training_state(best_psnr);
        LOG_INFO("Resumed training from epoch ", start_epoch,
                 ", best_psnr=", best_psnr, ", global_step=", global_step_);
    }

    // 打印训练配置摘要
    LOG_INFO("=== Training Configuration ===");
    LOG_INFO("  HR size: ", config_.hr_size, "x", config_.hr_size);
    LOG_INFO("  LR size: ", config_.lr_size, "x", config_.lr_size);
    LOG_INFO("  Batch size: ", config_.batch_size);
    LOG_INFO("  Epochs: ", config_.num_epochs);
    LOG_INFO("  Learning rate (G): ", config_.lr_g);
    LOG_INFO("  Learning rate (D): ", config_.lr_d);
    if (config_.use_attention) LOG_INFO("  CBAM attention: enabled");
    if (config_.use_spectral_norm) LOG_INFO("  Spectral norm: enabled");
    if (config_.frequency_weight > 0) LOG_INFO("  Frequency loss weight: ", config_.frequency_weight);
    if (config_.gradient_weight > 0) LOG_INFO("  Gradient loss weight: ", config_.gradient_weight);
    if (config_.r1_weight > 0) LOG_INFO("  R1 penalty weight: ", config_.r1_weight);
    if (start_epoch > 0) {
        LOG_INFO("  Resuming from epoch: ", start_epoch);
    }

    // ========================================================================
    // 主训练循环
    // ========================================================================
    for (int epoch = start_epoch; epoch < config_.num_epochs; ++epoch) {
        // 检查是否收到停止请求
        if (isStopRequested()) {
            LOG_INFO("正在保存训练状态...");
            save_checkpoint(epoch, "interrupted");
            save_training_state(epoch, best_psnr);
            LOG_INFO("训练已安全停止。使用 --resume ", config_.checkpoint_dir,
                     " 可从epoch ", epoch, " 继续训练");
            break;
        }

        // 训练一个epoch
        double epoch_loss = train_epoch(epoch);
        LOG_INFO("Epoch [", epoch, "/", config_.num_epochs, "] Average Loss: ", epoch_loss);

        // 如果在epoch训练中收到停止请求，保存并退出
        if (isStopRequested()) {
            LOG_INFO("正在保存训练状态...");
            save_checkpoint(epoch, "interrupted");
            save_training_state(epoch + 1, best_psnr);  // 下次从下一个epoch开始
            LOG_INFO("训练已安全停止。使用 --resume ", config_.checkpoint_dir,
                     " 可从epoch ", epoch + 1, " 继续训练");
            break;
        }

        // 定期验证
        if ((epoch + 1) % config_.val_interval == 0) {
            double psnr = validate(epoch);

            // 如果是最佳PSNR，保存best检查点
            if (psnr > best_psnr) {
                best_psnr = psnr;
                save_checkpoint(epoch, "best");
                save_training_state(epoch + 1, best_psnr);  // 同步保存训练状态
                LOG_INFO("New best PSNR: ", best_psnr, " dB");
            }
        }

        // 定期保存检查点
        if ((epoch + 1) % config_.save_interval == 0) {
            save_checkpoint(epoch);
            save_training_state(epoch + 1, best_psnr);  // 同步保存训练状态
        }
    }

    if (!isStopRequested()) {
        LOG_INFO("Training completed! Best PSNR: ", best_psnr, " dB");
    }
}

}  // namespace facesr
