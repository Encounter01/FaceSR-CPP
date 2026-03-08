/**
 * @file trainer.cpp
 * @brief 训练器类实现
 */

#include "trainer.h"
#include "utils/image_utils.h"
#include "common/logger.h"
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <cmath>

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

torch::Device Trainer::initDevice() {
    switch (config_.device_type) {
        case DeviceType::CPU:
            LOG_INFO("Device mode: CPU");
            return torch::kCPU;
        case DeviceType::CUDA:
            if (!torch::cuda::is_available()) {
                LOG_WARN("CUDA requested but not available, falling back to CPU");
                return torch::kCPU;
            }
            LOG_INFO("Device mode: GPU (CUDA)");
            return torch::kCUDA;
        case DeviceType::Hybrid:
            if (!torch::cuda::is_available()) {
                LOG_WARN("Hybrid mode requires CUDA, falling back to CPU");
                config_.device_type = DeviceType::CPU;
                return torch::kCPU;
            }
            LOG_INFO("Device mode: GPU+CPU Hybrid (GPU training, CPU data loading)");
            LOG_INFO("  pin_memory: ", (config_.pin_memory ? "enabled" : "disabled"));
            LOG_INFO("  data loader threads: ", config_.num_workers);
            return torch::kCUDA;
        case DeviceType::Auto:
        default:
            if (torch::cuda::is_available()) {
                LOG_INFO("Device mode: GPU+CPU Hybrid (auto)");
                config_.device_type = DeviceType::Hybrid;
                config_.pin_memory = true;
                return torch::kCUDA;
            }
            LOG_INFO("Device mode: CPU (auto)");
            return torch::kCPU;
    }
}

Trainer::Trainer(const TrainConfig& config)
    : config_(config),
      device_(initDevice()) {

    LOG_INFO("Using device: ", (device_.is_cuda() ? "CUDA" : "CPU"));

    // 创建目录
    fs::create_directories(config_.checkpoint_dir);
    fs::create_directories(config_.result_dir);

    // 构建生成器
    generator_ = models::RRDBNet(3, 3, constants::RRDB_NUM_FEATURES,
                                  constants::RRDB_NUM_BLOCKS,
                                  constants::RRDB_GROWTH_CHANNELS,
                                  config_.scale);
    generator_->to(device_);

    // 构建判别器
    discriminator_ = models::VGGStyleDiscriminator(3, 64, config_.hr_size);
    discriminator_->to(device_);

    // 打印参数量
    LOG_INFO("Generator parameters: ", generator_->get_num_parameters() / 1e6, "M");
    LOG_INFO("Discriminator parameters: ", discriminator_->get_num_parameters() / 1e6, "M");

    // 构建损失函数
    loss_fn_ = std::make_shared<models::CombinedLoss>(
        config_.pixel_weight,
        config_.perceptual_weight,
        config_.gan_weight,
        config_.pixel_loss_type,
        config_.gan_type
    );
    loss_fn_->to(device_);

    // 加载VGG权重(如果提供)
    if (!config_.vgg_weights_path.empty() && fs::exists(config_.vgg_weights_path)) {
        loss_fn_->load_vgg_weights(config_.vgg_weights_path);
    }

    // 构建优化器
    optimizer_g_ = std::make_unique<torch::optim::Adam>(
        generator_->parameters(),
        torch::optim::AdamOptions(config_.lr_g).betas({0.9, 0.99})
    );

    optimizer_d_ = std::make_unique<torch::optim::Adam>(
        discriminator_->parameters(),
        torch::optim::AdamOptions(config_.lr_d).betas({0.9, 0.99})
    );

    // 构建数据加载器
    // CPU负责数据加载和预处理, 通过多线程并行化
    auto train_dataset = utils::FaceSRDataset(
        config_.train_hr_dir,
        config_.train_lr_dir,
        config_.hr_size,
        config_.lr_size,
        true  // augment
    ).map(torch::data::transforms::Stack<>());

    train_loader_ = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(config_.batch_size)
            .workers(config_.num_workers)
    );

    LOG_INFO("Data loader initialized");
    if (config_.device_type == DeviceType::Hybrid) {
        LOG_INFO("Hybrid mode: CPU threads for data loading, GPU for training");
        LOG_INFO("  data loader threads: ", config_.num_workers);
        LOG_INFO("  pin_memory: ", (config_.pin_memory ? "enabled" : "disabled"));
    }
}

std::tuple<bool, bool> Trainer::get_training_phase(int epoch) {
    if (epoch < config_.phase1_epochs) {
        return {false, false};   // Phase1: pixel loss only
    } else if (epoch < config_.phase2_epochs) {
        return {true, false};    // Phase2: pixel + perceptual loss
    } else {
        return {true, true};     // Phase3: all losses
    }
}

double Trainer::get_current_lr() const {
    if (!optimizer_g_->param_groups().empty()) {
        return optimizer_g_->param_groups()[0].options().get_lr();
    }
    return config_.lr_g;
}

void Trainer::update_learning_rate(int epoch) {
    if (config_.lr_scheduler_type == LRSchedulerType::None) {
        return;
    }

    double new_lr_g = config_.lr_g;
    double new_lr_d = config_.lr_d;
    const int total_epochs = config_.num_epochs;

    // Warmup: linear ramp-up
    if (config_.lr_warmup_epochs > 0 && epoch < config_.lr_warmup_epochs) {
        double warmup_factor = static_cast<double>(epoch + 1) / config_.lr_warmup_epochs;
        new_lr_g = config_.lr_g * warmup_factor;
        new_lr_d = config_.lr_d * warmup_factor;
    } else {
        int effective_epoch = epoch - config_.lr_warmup_epochs;
        int effective_total = total_epochs - config_.lr_warmup_epochs;

        switch (config_.lr_scheduler_type) {
            case LRSchedulerType::CosineAnnealing: {
                // lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + cos(pi * epoch / total))
                double cosine = std::cos(M_PI * effective_epoch / std::max(effective_total, 1));
                new_lr_g = config_.lr_min + 0.5 * (config_.lr_g - config_.lr_min) * (1.0 + cosine);
                new_lr_d = config_.lr_min + 0.5 * (config_.lr_d - config_.lr_min) * (1.0 + cosine);
                break;
            }
            case LRSchedulerType::Step: {
                int num_decays = effective_epoch / std::max(config_.lr_decay_step, 1);
                double factor = std::pow(config_.lr_decay_gamma, num_decays);
                new_lr_g = std::max(config_.lr_g * factor, config_.lr_min);
                new_lr_d = std::max(config_.lr_d * factor, config_.lr_min);
                break;
            }
            case LRSchedulerType::MultiStep: {
                int milestones[] = {
                    effective_total / 4,
                    effective_total / 2,
                    effective_total * 3 / 4,
                    effective_total * 7 / 8
                };
                int num_decays = 0;
                for (int m : milestones) {
                    if (effective_epoch >= m) num_decays++;
                }
                double factor = std::pow(config_.lr_decay_gamma, num_decays);
                new_lr_g = std::max(config_.lr_g * factor, config_.lr_min);
                new_lr_d = std::max(config_.lr_d * factor, config_.lr_min);
                break;
            }
            default:
                return;
        }
    }

    for (auto& group : optimizer_g_->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(new_lr_g);
    }
    for (auto& group : optimizer_d_->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(new_lr_d);
    }
}

double Trainer::train_epoch(int epoch) {
    generator_->train();
    discriminator_->train();

    auto [use_perceptual, use_gan] = get_training_phase(epoch);
    last_epoch_interrupted_ = false;

    utils::AverageMeter loss_meter;
    int batch_idx = 0;

    for (auto& batch : *train_loader_) {
        if (isStopRequested()) {
            LOG_INFO("Stop requested, finishing current epoch safely...");
            last_epoch_interrupted_ = true;
            break;
        }

        bool non_blocking = (config_.device_type == DeviceType::Hybrid);
        auto lr = batch.data.to(device_, non_blocking);
        auto hr = batch.target.to(device_, non_blocking);

        if (use_gan) {
            optimizer_d_->zero_grad();

            torch::Tensor sr;
            {
                torch::NoGradGuard no_grad;
                sr = generator_->forward(lr);
            }

            auto pred_real = discriminator_->forward(hr);
            auto loss_d_real = models::GANLoss(config_.gan_type).forward(pred_real, true, true);

            auto pred_fake = discriminator_->forward(sr.detach());
            auto loss_d_fake = models::GANLoss(config_.gan_type).forward(pred_fake, false, true);

            auto loss_d = (loss_d_real + loss_d_fake) / 2;
            loss_d.backward();
            optimizer_d_->step();
        }

        optimizer_g_->zero_grad();

        auto sr = generator_->forward(lr);

        torch::Tensor disc_pred_fake;
        if (use_gan) {
            disc_pred_fake = discriminator_->forward(sr);
        }

        auto losses = loss_fn_->forward(sr, hr, disc_pred_fake, use_perceptual, use_gan);
        auto loss_g = losses["total"];

        loss_g.backward();
        optimizer_g_->step();

        loss_meter.update(loss_g.item<double>(), lr.size(0));

        if (batch_idx % config_.log_interval == 0) {
            LOG_PROGRESS("Epoch ", epoch, " [", batch_idx, "] Loss: ",
                        std::fixed, std::setprecision(4), loss_meter.avg());
        }

        batch_idx++;
        global_step_++;
    }

    LOG_END_PROGRESS();
    return loss_meter.avg();
}

double Trainer::validate(int epoch) {
    generator_->eval();

    utils::MetricCalculator metrics;

    auto val_dataset = utils::FaceSRDataset(
        config_.val_hr_dir,
        config_.val_lr_dir,
        config_.hr_size,
        config_.lr_size,
        false  // no augment
    ).map(torch::data::transforms::Stack<>());

    auto val_loader = torch::data::make_data_loader(
        std::move(val_dataset),
        torch::data::DataLoaderOptions().batch_size(1)
    );

    int batch_idx = 0;
    for (auto& batch : *val_loader) {
        bool non_blocking = (config_.device_type == DeviceType::Hybrid);
        auto lr = batch.data.to(device_, non_blocking);
        auto hr = batch.target.to(device_, non_blocking);

        torch::Tensor sr;
        {
            torch::NoGradGuard no_grad;
            sr = generator_->forward(lr);
            sr = sr.clamp(0, 1);
        }

        metrics.update(sr, hr);

        // Save first validation image
        if (batch_idx == 0) {
            utils::save_tensor_image(sr, config_.result_dir + "/val_epoch" + std::to_string(epoch) + "_sr.png");
        }

        batch_idx++;
    }

    double avg_psnr = metrics.get_avg_psnr();
    double avg_ssim = metrics.get_avg_ssim();

    LOG_INFO("Validation - PSNR: ", std::fixed, std::setprecision(2), avg_psnr,
             " dB, SSIM: ", std::setprecision(4), avg_ssim);

    return avg_psnr;
}

void Trainer::save_checkpoint(int epoch, bool is_best, bool save_epoch_snapshot) {
    try {
        if (save_epoch_snapshot) {
            torch::save(generator_, config_.checkpoint_dir + "/generator_epoch" + std::to_string(epoch) + ".pt");
            torch::save(discriminator_, config_.checkpoint_dir + "/discriminator_epoch" + std::to_string(epoch) + ".pt");
        }

        torch::save(generator_, config_.checkpoint_dir + "/generator_latest.pt");
        torch::save(discriminator_, config_.checkpoint_dir + "/discriminator_latest.pt");

        if (is_best) {
            torch::save(generator_, config_.checkpoint_dir + "/generator_best.pt");
            torch::save(discriminator_, config_.checkpoint_dir + "/discriminator_best.pt");
            LOG_INFO("Saved best model (Epoch ", epoch, ")");
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to save checkpoint: ", e.what());
    }
}

void Trainer::load_checkpoint(const std::string& path) {
    try {
        std::string gen_path, disc_path;

        if (fs::is_directory(path)) {
            gen_path = path + "/generator_latest.pt";
            disc_path = path + "/discriminator_latest.pt";
        } else {
            gen_path = path;
            disc_path = fs::path(path).parent_path().string() + "/discriminator_latest.pt";
        }

        LOG_INFO("Loading generator: ", gen_path);
        torch::load(generator_, gen_path);
        generator_->to(device_);

        if (fs::exists(disc_path)) {
            LOG_INFO("Loading discriminator: ", disc_path);
            torch::load(discriminator_, disc_path);
            discriminator_->to(device_);
        } else {
            LOG_WARN("Discriminator checkpoint not found: ", disc_path);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load checkpoint: ", e.what());
        throw;
    }
}

void Trainer::save_training_state(int epoch, double best_psnr) {
    // Save training state to binary file
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

    // Save optimizer states
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

int Trainer::load_training_state(double& best_psnr) {
    std::string state_path = config_.checkpoint_dir + "/train_state.bin";
    if (!fs::exists(state_path)) {
        LOG_WARN("Training state file not found: ", state_path);
        // Fallback: infer epoch from checkpoint filenames
        int max_epoch = -1;
        for (auto& entry : fs::directory_iterator(config_.checkpoint_dir)) {
            std::string name = entry.path().filename().string();
            // Match generator_epochN.pt
            if (name.rfind("generator_epoch", 0) == 0 && name.size() >= 19) {
                try {
                    int ep = std::stoi(name.substr(15, name.size() - 18));
                    if (ep > max_epoch) max_epoch = ep;
                } catch (...) {}
            }
        }
        if (max_epoch >= 0) {
            int resume_epoch = max_epoch + 1;
            LOG_INFO("Inferred resume epoch from checkpoint files: ", resume_epoch);
            return resume_epoch;
        }
        // No epoch snapshots found, check if latest exists (resume from epoch 1)
        if (fs::exists(config_.checkpoint_dir + "/generator_latest.pt")) {
            LOG_INFO("Found latest checkpoint but no epoch info, resuming from epoch 1");
            return 1;
        }
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

    // Load optimizer states
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
                 " - using fresh optimizers");
    }

    return epoch;
}

bool Trainer::has_latest_checkpoint(const std::string& checkpoint_dir) const {
    const fs::path checkpoint_path(checkpoint_dir);
    return fs::exists(checkpoint_path / "generator_latest.pt");
}

void Trainer::train(const std::string& resume_path) {
    int start_epoch = 0;
    double best_psnr = 0;
    std::string actual_resume_path = resume_path;

    if (actual_resume_path.empty() && config_.auto_resume && has_latest_checkpoint(config_.checkpoint_dir)) {
        actual_resume_path = config_.checkpoint_dir;
        LOG_INFO("Detected existing latest checkpoint. Auto resume from: ", actual_resume_path);
    }

    if (!actual_resume_path.empty()) {
        if (fs::exists(actual_resume_path)) {
            fs::path checkpoint_root = fs::is_directory(actual_resume_path)
                ? fs::path(actual_resume_path)
                : fs::path(actual_resume_path).parent_path();
            if (!checkpoint_root.empty()) {
                config_.checkpoint_dir = checkpoint_root.string();
            }
        }

        load_checkpoint(actual_resume_path);
        start_epoch = load_training_state(best_psnr);
        LOG_INFO("Resumed training from epoch ", start_epoch,
                 ", best_psnr=", best_psnr, ", global_step=", global_step_);
    }

    LOG_INFO("Starting training...");
    LOG_INFO("  - Epochs: ", config_.num_epochs);
    LOG_INFO("  - Batch size: ", config_.batch_size);
    LOG_INFO("  - Learning rate: ", config_.lr_g);
    LOG_INFO("  - LR scheduler: ", lrSchedulerTypeToString(config_.lr_scheduler_type));
    if (config_.lr_scheduler_type != LRSchedulerType::None) {
        LOG_INFO("  - LR min: ", config_.lr_min);
        if (config_.lr_warmup_epochs > 0) {
            LOG_INFO("  - LR warmup epochs: ", config_.lr_warmup_epochs);
        }
    }
    LOG_INFO("  - Validation interval: ", config_.val_interval, " epochs");
    LOG_INFO("  - Pixel loss type: ", pixelLossTypeToString(config_.pixel_loss_type));
    LOG_INFO("  - GAN type: ", ganTypeToString(config_.gan_type));
    LOG_INFO("  - Save latest: every ", config_.latest_save_interval, " epoch(s)");
    LOG_INFO("  - Save snapshot: every ", config_.save_interval, " epoch(s)");
    LOG_INFO("  - Phase1 (pixel only): epoch 0-", config_.phase1_epochs - 1);
    LOG_INFO("  - Phase2 (pixel+perceptual): epoch ", config_.phase1_epochs, "-", config_.phase2_epochs - 1);
    LOG_INFO("  - Phase3 (full GAN): epoch ", config_.phase2_epochs, "+");
    if (start_epoch > 0) {
        LOG_INFO("  - Resuming from epoch: ", start_epoch);
    }

    // Print upcoming save schedule
    LOG_INFO("--- Save schedule (next 20 events) ---");
    int printed = 0;
    for (int ep = start_epoch; ep < config_.num_epochs && printed < 20; ++ep) {
        bool will_val = (config_.val_interval > 0 && ep % config_.val_interval == 0);
        bool will_snap = (config_.save_interval > 0 && ep % config_.save_interval == 0);
        if (will_val || will_snap) {
            std::string actions;
            if (will_val) actions += "validate";
            if (will_snap) {
                if (!actions.empty()) actions += " + ";
                actions += "snapshot";
            }
            LOG_INFO("  Epoch ", ep, ": ", actions);
            printed++;
        }
    }
    LOG_INFO("--------------------------------------");

    for (int epoch = start_epoch; epoch < config_.num_epochs; ++epoch) {
        if (isStopRequested()) {
            LOG_INFO("Saving training state before stop...");
            save_checkpoint(epoch, false, false);
            save_training_state(epoch, best_psnr);
            LOG_INFO("Training stopped safely. Use --resume ", config_.checkpoint_dir,
                     " to continue from epoch ", epoch);
            break;
        }

        // Update learning rate
        update_learning_rate(epoch);

        double train_loss = train_epoch(epoch);
        LOG_INFO("Epoch ", epoch, " - Loss: ", std::fixed, std::setprecision(4), train_loss,
                 ", LR: ", std::scientific, std::setprecision(2), get_current_lr());

        if (isStopRequested()) {
            const int resume_epoch = last_epoch_interrupted_ ? epoch : (epoch + 1);
            LOG_INFO("Saving training state before stop...");
            save_checkpoint(epoch, false, false);
            save_training_state(resume_epoch, best_psnr);
            LOG_INFO("Training stopped safely. Use --resume ", config_.checkpoint_dir,
                     " to continue from epoch ", resume_epoch);
            break;
        }

        bool is_best = false;
        if (config_.val_interval > 0 && epoch % config_.val_interval == 0) {
            double val_psnr = validate(epoch);
            if (val_psnr > best_psnr) {
                best_psnr = val_psnr;
                is_best = true;
            }
        }

        int latest_save_interval = config_.latest_save_interval;
        if (latest_save_interval <= 0) {
            latest_save_interval = 1;
        }

        const bool save_latest_state = (epoch % latest_save_interval == 0);
        const bool save_epoch_snapshot =
            (config_.save_interval > 0) && (epoch % config_.save_interval == 0);

        if (save_latest_state || save_epoch_snapshot || is_best) {
            std::string save_info = "Epoch " + std::to_string(epoch) + " saving:";
            if (save_latest_state) save_info += " [latest]";
            if (save_epoch_snapshot) save_info += " [snapshot]";
            if (is_best) save_info += " [best]";
            LOG_INFO(save_info);
            save_checkpoint(epoch, is_best, save_epoch_snapshot);
            save_training_state(epoch + 1, best_psnr);
        }
    }

    if (!isStopRequested()) {
        LOG_INFO("Training completed! Best PSNR: ", std::fixed, std::setprecision(2), best_psnr, " dB");
    }
}

}  // namespace facesr
