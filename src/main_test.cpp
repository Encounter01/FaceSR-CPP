// ============================================================================
// main_test.cpp — 推理程序入口
//
// 命令行用法:
//   facesr_test --model checkpoints/final/facesr_a4_best_psnr28.6019.pt --input photo.jpg --output result.jpg
//   facesr_test --model model.pt --input input_folder/ --output output_folder/
//   facesr_test --model model.pt --input photo.jpg --scale 4 --cpu
//
// 入口职责：
// - 加载训练好的生成器权重。
// - 判断输入是单张图像还是目录。
// - 调用 Inference 完成预处理、模型前向和结果保存。
// ============================================================================

#include "inference.h"
#include "common/logger.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// 打印命令行帮助信息
void print_usage() {
    std::cout << "FaceSR Inference Tool\n"
              << "Usage: facesr_test [options]\n"
              << "\nOptions:\n"
              << "  --model <path>    Path to trained generator model (.pt)\n"
              << "  --input <path>    Input image file or directory\n"
              << "  --output <path>   Output file or directory (default: input_sr.ext)\n"
              << "  --scale <n>       Super-resolution scale factor (default: 4)\n"
              << "  --cpu             Force CPU inference\n"
              << "  --attention       Enable CBAM attention (must match training)\n"
              << "  -h, --help        Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "";
    std::string input_path = "";
    std::string output_path = "";
    int scale = 4;
    bool use_gpu = true;
    bool use_attention = false;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
        else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
        else if (arg == "--scale" && i + 1 < argc) {
            scale = std::stoi(argv[++i]);
        }
        else if (arg == "--cpu") {
            use_gpu = false;
        }
        else if (arg == "--attention") {
            use_attention = true;
        }
    }

    // 检查必需参数
    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        print_usage();
        return 1;
    }
    if (input_path.empty()) {
        std::cerr << "Error: --input is required\n";
        print_usage();
        return 1;
    }

    try {
        // 创建推理器实例（加载模型权重）。
        // --attention 必须和训练时的 use_attention 一致，否则 LibTorch 原生权重形状会不匹配。
        LOG_INFO("Loading model: ", model_path);
        facesr::Inference inferencer(model_path, scale, use_gpu, use_attention);

        if (!inferencer.isModelLoaded()) {
            LOG_FATAL("Failed to load model!");
            return 1;
        }

        if (fs::is_regular_file(input_path)) {
            // 输入是单个文件
            // 如果未指定输出路径，自动生成: 原名_sr.ext
            if (output_path.empty()) {
                auto p = fs::path(input_path);
                output_path = (p.parent_path() /
                               (p.stem().string() + "_sr" + p.extension().string())).string();
            }
            if (!inferencer.process_file(input_path, output_path)) {
                return 1;
            }
        }
        else if (fs::is_directory(input_path)) {
            // 输入是目录，批量处理
            if (output_path.empty()) {
                output_path = input_path + "_sr";  // 默认输出到 input_dir_sr/
            }
            int count = inferencer.process_folder(input_path, output_path);
            LOG_INFO("Processed ", count, " images");
            if (count == 0) {
                return 1;
            }
        }
        else {
            LOG_ERROR("Input path does not exist: ", input_path);
            return 1;
        }
    } catch (const std::exception& e) {
        LOG_FATAL("Inference failed: ", e.what());
        return 1;
    }

    return 0;
}
