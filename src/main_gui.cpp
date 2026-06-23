/**
 * @file main_gui.cpp
 * @brief GUI程序入口
 *
 * GUI 只是交互入口：模型加载、图像选择、后台推理和保存结果都在 MainWindow 中组织。
 * 算法实现仍复用 Inference 和 RRDBNet，避免 GUI 直接耦合训练代码。
 */

#include <QApplication>
#include "gui/main_window.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // 设置应用信息
    app.setApplicationName("FaceSR");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("XSYU");

    // 设置应用样式
    app.setStyle("Fusion");

    // 创建并显示主窗口
    facesr::gui::MainWindow window;
    window.show();

    return app.exec();
}
