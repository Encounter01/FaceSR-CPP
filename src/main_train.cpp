// ============================================================================
// main_train.cpp — 训练程序入口
//
// 命令行用法:
//   facesr_train --config config/train_config.ini
//   facesr_train --train-hr data/train/hr --batch-size 8 --epochs 100
//   facesr_train --resume checkpoints              (从目录恢复)
//   facesr_train --resume checkpoints/legacy/generator_latest.pt  (从文件恢复)
//
// 入口职责：
// - 解析命令行或 INI 配置，得到 TrainConfig。
// - 注册 Ctrl+C 安全中断，让 Trainer 有机会保存状态。
// - 真正的训练逻辑在 Trainer 中，这里只负责组装和启动。
// ============================================================================

#include "trainer.h"
#include "common/logger.h"
#include <iostream>
#include <string>
#include <csignal>

#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================================
// 信号处理 — 优雅停止训练
// ============================================================================
static volatile bool force_exit = false;

void signal_handler(int sig) {
    if (force_exit) {
        // 第二次Ctrl+C，强制退出
        LOG_WARN("强制退出！");
        std::_Exit(1);
    }
    force_exit = true;
    facesr::Trainer::requestStop();
    LOG_INFO("收到停止信号，正在保存训练状态... (再次按Ctrl+C强制退出)");
}

#ifdef _WIN32
BOOL WINAPI console_handler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_CLOSE_EVENT) {
        if (force_exit) {
            LOG_WARN("强制退出！");
            std::_Exit(1);
        }
        force_exit = true;
        facesr::Trainer::requestStop();
        LOG_INFO("收到停止信号，正在保存训练状态... (再次按Ctrl+C强制退出)");
        return TRUE;
    }
    return FALSE;
}
#endif

// 打印命令行帮助信息
void print_usage() {
    std::cout << "FaceSR Training Tool\n"
              << "Usage: facesr_train [options]\n"
              << "\nOptions:\n"
              << "  --config <path>     Load config from INI file\n"
              << "  --train-hr <path>   Training HR image directory\n"
              << "  --train-lr <path>   Training LR image directory (optional)\n"
              << "  --val-hr <path>     Validation HR image directory\n"
              << "  --batch-size <n>    Batch size (default: 12)\n"
              << "  --epochs <n>        Number of epochs (default: 300)\n"
              << "  --lr <rate>         Learning rate (default: 0.0002)\n"
              << "  --resume <path>     Resume from checkpoint (file or directory)\n"
              << "  --cpu               Force CPU training\n"
              << "  -h, --help          Show this help\n";
}

int main(int argc, char* argv[]) {
    // 注册信号处理器（优雅停止）
    std::signal(SIGINT, signal_handler);
#ifdef _WIN32
    SetConsoleCtrlHandler(console_handler, TRUE);
#endif

    facesr::TrainConfig config;   // 创建默认训练配置
    std::string resume_path = ""; // 恢复训练的检查点路径

    // 解析命令行参数。
    // 命令行参数优先级高于默认 TrainConfig；--config 可一次性加载论文实验配置。
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
        else if (arg == "--config" && i + 1 < argc) {
            // 从INI配置文件加载所有参数
            config.loadFromFile(argv[++i]);
        }
        else if (arg == "--train-hr" && i + 1 < argc) {
            config.train_hr_dir = argv[++i];
        }
        else if (arg == "--train-lr" && i + 1 < argc) {
            config.train_lr_dir = argv[++i];
        }
        else if (arg == "--val-hr" && i + 1 < argc) {
            config.val_hr_dir = argv[++i];
        }
        else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            config.num_epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--lr" && i + 1 < argc) {
            config.lr_g = std::stod(argv[++i]);
            config.lr_d = config.lr_g;  // 生成器和判别器使用相同学习率
        }
        else if (arg == "--resume" && i + 1 < argc) {
            resume_path = argv[++i];
        }
        else if (arg == "--cpu") {
            config.device_type = facesr::DeviceType::CPU;
        }
    }

    // 打印训练配置
    LOG_INFO("=== FaceSR Training ===");
    LOG_INFO("Training HR dir: ", config.train_hr_dir);
    LOG_INFO("Validation HR dir: ", config.val_hr_dir);
    LOG_INFO("Batch size: ", config.batch_size);
    LOG_INFO("Epochs: ", config.num_epochs);
    LOG_INFO("Learning rate: ", config.lr_g);

    // 创建 Trainer 实例并开始训练。
    // resume_path 可以是 checkpoint 文件或目录，具体恢复策略由 Trainer::load_checkpoint 处理。
    try {
        facesr::Trainer trainer(config);
        trainer.train(resume_path);
    } catch (const std::exception& e) {
        LOG_FATAL("Training failed: ", e.what());
        return 1;
    }

    return 0;
}
