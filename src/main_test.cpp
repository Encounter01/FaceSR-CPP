/**
 * @file main_test.cpp
 * @brief 测试/推理程序入口 (支持GPU+CPU混合模式)
 */

#include "inference.h"
#include <iostream>
#include <string>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

/**
 * @brief 解析相对路径: 从当前目录向上搜索, 找到包含该路径的祖先目录
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
              << "  --model <路径>        模型文件路径 (必需)\n"
              << "  --input <路径>        输入图像或文件夹路径 (必需)\n"
              << "  --output <路径>       输出路径 (默认: results)\n"
              << "  --scale <数值>        放大倍数 (默认: 4)\n"
              << "  --device <类型>       设备模式: auto/hybrid/gpu/cpu (默认: auto)\n"
              << "                          auto   - 自动选择最优模式\n"
              << "                          hybrid - GPU+CPU混合流水线 (推荐)\n"
              << "                          gpu    - 仅GPU\n"
              << "                          cpu    - 仅CPU\n"
              << "  --cpu                 仅使用CPU (等同于 --device cpu)\n"
              << "  --help                显示此帮助信息\n"
              << "\n"
              << "GPU+CPU混合模式说明:\n"
              << "  在混合模式下, 程序同时利用GPU和CPU:\n"
              << "  - CPU线程: 负责图像加载、预处理、后处理、保存 (IO密集型)\n"
              << "  - GPU线程: 负责神经网络推理 (计算密集型)\n"
              << "  - 三阶段流水线并行, 最大化硬件利用率\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::string model_path;
    std::string input_path;
    std::string output_path = "results";
    int scale = 4;
    facesr::DeviceType device_type = facesr::DeviceType::Auto;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--scale" && i + 1 < argc) {
            scale = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            std::string dev = argv[++i];
            if (dev == "auto") {
                device_type = facesr::DeviceType::Auto;
            } else if (dev == "hybrid") {
                device_type = facesr::DeviceType::Hybrid;
            } else if (dev == "gpu" || dev == "cuda") {
                device_type = facesr::DeviceType::CUDA;
            } else if (dev == "cpu") {
                device_type = facesr::DeviceType::CPU;
            } else {
                std::cerr << "未知设备类型: " << dev << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--cpu") {
            device_type = facesr::DeviceType::CPU;
        } else {
            std::cerr << "未知参数: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // 检查必需参数
    if (model_path.empty()) {
        std::cerr << "错误: 请指定模型路径 (--model)" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (input_path.empty()) {
        std::cerr << "错误: 请指定输入路径 (--input)" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "超分辨率人脸图像重建 - 推理程序 (C++)" << std::endl;
    std::cout << "支持GPU+CPU混合流水线加速" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        facesr::Inference inferencer(model_path, scale, device_type);

        if (!inferencer.is_loaded()) {
            std::cerr << "模型加载失败" << std::endl;
            return 1;
        }

        fs::path input(input_path);

        if (fs::is_regular_file(input)) {
            // 处理单个文件
            fs::path out_path(output_path);
            if (fs::is_directory(out_path)) {
                out_path = out_path / (input.stem().string() + "_sr" + input.extension().string());
            }
            inferencer.process_file(input_path, out_path.string());
        } else if (fs::is_directory(input)) {
            // 批量处理文件夹 (混合模式下自动启用流水线)
            inferencer.process_folder(input_path, output_path);
        } else {
            std::cerr << "错误: 输入路径不存在: " << input_path << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
