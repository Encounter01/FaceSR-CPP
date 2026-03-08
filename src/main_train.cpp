/**
 * @file main_train.cpp
 * @brief 训练程序入口 (支持GPU+CPU混合模式)
 */

#include "trainer.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <csignal>
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

// ============================================================================
// 信号处理 — 优雅停止训练
// ============================================================================
static volatile bool force_exit = false;

void signal_handler(int sig) {
    if (force_exit) {
        std::cerr << "\n强制退出！" << std::endl;
        std::_Exit(1);
    }
    force_exit = true;
    facesr::Trainer::requestStop();
    std::cerr << "\n收到停止信号，正在保存训练状态... (再次按Ctrl+C强制退出)" << std::endl;
}

#ifdef _WIN32
BOOL WINAPI console_handler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_CLOSE_EVENT) {
        if (force_exit) {
            std::cerr << "\n强制退出！" << std::endl;
            std::_Exit(1);
        }
        force_exit = true;
        facesr::Trainer::requestStop();
        std::cerr << "\n收到停止信号，正在保存训练状态... (再次按Ctrl+C强制退出)" << std::endl;
        return TRUE;
    }
    return FALSE;
}
#endif

/**
 * @brief 解析相对路径: 从当前目录向上搜索, 找到包含该路径的祖先目录
 *
 * 解决从 build/bin/Release/ 运行时找不到 data/ 目录的问题
 */
std::string resolveRelativePath(const std::string& rel_path) {
    if (rel_path.empty() || fs::path(rel_path).is_absolute()) return rel_path;
    if (fs::exists(rel_path)) return rel_path;

    auto dir = fs::current_path();
    for (int i = 0; i < 5; ++i) {
        dir = dir.parent_path();
        auto candidate = dir / rel_path;
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    return rel_path;
}

void print_usage(const char* program_name) {
    std::cout << "用法: " << program_name << " [选项]\n"
              << "\n"
              << "选项:\n"
              << "  --train-hr <路径>     训练集HR图像目录 (默认: data/train/HR)\n"
              << "  --train-lr <路径>     训练集LR图像目录 (默认: data/train/LR)\n"
              << "  --val-hr <路径>       验证集HR图像目录 (默认: data/val/HR)\n"
              << "  --val-lr <路径>       验证集LR图像目录 (默认: data/val/LR)\n"
              << "  --batch-size <数值>   批量大小 (默认: 12)\n"
              << "  --epochs <数值>       训练轮数 (默认: 300)\n"
              << "  --lr <数值>           学习率 (默认: 0.0002)\n"
              << "  --workers <数值>      数据加载线程数 (默认: 4)\n"
              << "  --device <类型>       设备模式: auto/hybrid/gpu/cpu (默认: auto)\n"
              << "                          auto   - 自动选择最优模式\n"
              << "                          hybrid - GPU训练+CPU数据加载 (推荐)\n"
              << "                          gpu    - 仅GPU\n"
              << "                          cpu    - 仅CPU\n"
              << "  --lr-scheduler <类型> 学习率调度器: cosine/step/multistep/none (默认: cosine)\n"
              << "  --no-pin-memory       禁用pin_memory (默认启用)\n"
              << "  --config <路径>       配置文件路径\n"
              << "  --resume <路径>       恢复训练的检查点目录\n"
              << "  --help                显示此帮助信息\n"
              << "\n"
              << "GPU+CPU混合模式说明:\n"
              << "  在混合模式下, 程序同时利用GPU和CPU:\n"
              << "  - CPU线程池: 负责数据加载、图像预处理、数据增强\n"
              << "  - GPU: 负责模型前向/反向传播、参数更新\n"
              << "  - 使用pin_memory加速CPU→GPU数据传输\n"
              << "  - 使用non_blocking传输实现CPU/GPU计算重叠\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // 注册信号处理器（优雅停止）
    std::signal(SIGINT, signal_handler);
#ifdef _WIN32
    SetConsoleCtrlHandler(console_handler, TRUE);
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    facesr::TrainConfig config;
    std::string resume_path;
    std::string config_path;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--train-hr" && i + 1 < argc) {
            config.train_hr_dir = argv[++i];
        } else if (arg == "--train-lr" && i + 1 < argc) {
            config.train_lr_dir = argv[++i];
        } else if (arg == "--val-hr" && i + 1 < argc) {
            config.val_hr_dir = argv[++i];
        } else if (arg == "--val-lr" && i + 1 < argc) {
            config.val_lr_dir = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.lr_g = std::stod(argv[++i]);
            config.lr_d = config.lr_g;
        } else if (arg == "--workers" && i + 1 < argc) {
            config.num_workers = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            std::string dev = argv[++i];
            if (dev == "auto") {
                config.device_type = facesr::DeviceType::Auto;
            } else if (dev == "hybrid") {
                config.device_type = facesr::DeviceType::Hybrid;
            } else if (dev == "gpu" || dev == "cuda") {
                config.device_type = facesr::DeviceType::CUDA;
            } else if (dev == "cpu") {
                config.device_type = facesr::DeviceType::CPU;
            } else {
                std::cerr << "未知设备类型: " << dev << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--lr-scheduler" && i + 1 < argc) {
            config.lr_scheduler_type = facesr::stringToLRSchedulerType(argv[++i]);
        } else if (arg == "--no-pin-memory") {
            config.pin_memory = false;
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--resume" && i + 1 < argc) {
            resume_path = argv[++i];
        } else {
            std::cerr << "未知参数: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // 自动搜索配置文件 (未指定 --config 时)
    if (config_path.empty()) {
        const std::string candidates[] = {
            "config/train_config.ini",
            "train_config.ini",
        };
        for (const auto& c : candidates) {
            std::string resolved = resolveRelativePath(c);
            if (fs::exists(resolved)) {
                config_path = resolved;
                break;
            }
        }
    } else {
        config_path = resolveRelativePath(config_path);
    }

    // 从配置文件加载
    if (!config_path.empty()) {
        if (config.loadFromFile(config_path)) {
            std::cout << "已加载配置文件: " << config_path << std::endl;
        } else {
            std::cerr << "警告: 无法加载配置文件: " << config_path << std::endl;
        }
    } else {
        std::cout << "未找到配置文件, 使用默认参数" << std::endl;
    }

    // 解析相对路径 (从CWD向上搜索项目根目录)
    config.train_hr_dir = resolveRelativePath(config.train_hr_dir);
    config.train_lr_dir = resolveRelativePath(config.train_lr_dir);
    config.val_hr_dir = resolveRelativePath(config.val_hr_dir);
    config.val_lr_dir = resolveRelativePath(config.val_lr_dir);
    config.checkpoint_dir = resolveRelativePath(config.checkpoint_dir);
    config.result_dir = resolveRelativePath(config.result_dir);

    // 检查训练数据是否存在
    if (!fs::exists(config.train_hr_dir) || fs::is_empty(config.train_hr_dir)) {
        std::cerr << "错误: 训练数据目录不存在或为空: " << config.train_hr_dir << std::endl;
        std::cerr << "请先准备数据: python prepare_data.py --ffhq <FFHQ目录> --celebahq <CelebA-HQ目录>" << std::endl;
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "超分辨率人脸图像重建 - 训练程序 (C++)" << std::endl;
    std::cout << "支持GPU+CPU混合训练加速" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "配置:" << std::endl;
    std::cout << "  训练集HR: " << config.train_hr_dir << std::endl;
    std::cout << "  批量大小: " << config.batch_size << std::endl;
    std::cout << "  训练轮数: " << config.num_epochs << std::endl;
    std::cout << "  学习率: " << config.lr_g << std::endl;
    std::cout << "  学习率调度: " << facesr::lrSchedulerTypeToString(config.lr_scheduler_type) << std::endl;
    std::cout << "  验证间隔: " << config.val_interval << " epochs" << std::endl;
    std::cout << "  数据加载线程: " << config.num_workers << std::endl;
    std::cout << "  pin_memory: " << (config.pin_memory ? "启用" : "禁用") << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        facesr::Trainer trainer(config);
        trainer.train(resume_path);
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
