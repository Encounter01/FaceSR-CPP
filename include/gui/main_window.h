#pragma once
/**
 * @file main_window.h
 * @brief Qt主窗口
 *
 * GUI 层只负责交互和展示：
 * - 打开图像、加载模型、启动重建、保存结果。
 * - 通过后台线程调用 Inference，避免模型推理阻塞 Qt 主线程。
 * - 不直接依赖 Trainer；训练和推理在工程结构上保持分离。
 */

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QProgressBar>
#include <QThread>
#include <QPixmap>
#include <QSize>
#include <QStringList>
#include <memory>

// 前向声明 - 完全避免包含 OpenCV 和 LibTorch 头文件
class QResizeEvent;

namespace facesr {
    class Inference;
}

namespace facesr {
namespace gui {
    class BackgroundWidget;
    class FrostedWidget;
    class StyleConfig;
}
}

namespace facesr {
namespace gui {

/**
 * @brief 后台处理线程
 * 使用 PIMPL 模式隔离实现细节
 *
 * 单张图像推理可能耗时较长，放到 QThread 中执行可以保持窗口响应。
 */
class ProcessThread : public QThread {
    Q_OBJECT

public:
    ProcessThread(Inference* inferencer, const QPixmap& inputPixmap);
    ~ProcessThread();
    void run() override;

signals:
    void finished(QPixmap result);
    void error(QString message);
    void progress(int value);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};


/**
 * @brief 批量处理后台线程
 *
 * 目录批处理同样放在后台线程中执行；当前进度条只表示忙碌状态，不表示精确百分比。
 */
class BatchThread : public QThread {
    Q_OBJECT

public:
    BatchThread(Inference* inferencer, const QStringList& inputFiles, const QString& outputDir);
    ~BatchThread();
    void run() override;

signals:
    void finished(int count);
    void error(QString message);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};


/**
 * @brief 图像显示标签
 */
class ImageLabel : public QLabel {
    Q_OBJECT

public:
    ImageLabel(const QString& text = "", QWidget* parent = nullptr);

    void setImage(const QPixmap& pixmap);
    void clearImage();
    void setMaxDisplaySize(const QSize& maxSize);

    QPixmap getPixmap() const { return pixmap_; }

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void updateDisplay();
    QPixmap pixmap_;
    QSize maxDisplaySize_;
};


/**
 * @brief 主窗口类
 *
 * MainWindow 是演示闭环的组织者：QPixmap 用于界面显示，真正推理前再转换为 cv::Mat。
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void openImage();
    void loadModel();
    void processImage();
    void saveImage();
    void batchProcess();

    void onProcessFinished(QPixmap result);
    void onProcessError(QString message);

    void onBatchFinished(int count);
    void onBatchError(QString message);

    void showAbout();

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void setupUI();
    void setupMenuBar();
    void updateButtonStates();
    void updateStatus(const QString& message);

    // UI配置相关
    void loadUIConfig();
    void setupBackground();
    void applyTheme();

    // UI组件
    BackgroundWidget* backgroundWidget_ = nullptr;
    FrostedWidget* toolbarPanel_ = nullptr;
    ImageLabel* inputLabel_;
    ImageLabel* outputLabel_;
    QLabel* inputInfo_;
    QLabel* outputInfo_;
    QLabel* modelStatus_;
    QPushButton* btnOpen_;
    QPushButton* btnLoadModel_;
    QPushButton* btnProcess_;
    QPushButton* btnSave_;
    QPushButton* btnBatch_;
    QProgressBar* progressBar_;

    // 数据 - 使用 QPixmap 避免 cv::Mat 依赖
    QPixmap inputPixmap_;
    QPixmap outputPixmap_;
    QString currentImagePath_;  // 保存当前图像路径用于处理

    std::unique_ptr<Inference> inferencer_;
    ProcessThread* processThread_ = nullptr;

    // 样式配置
    StyleConfig* styleConfig_ = nullptr;
};

}  // namespace gui
}  // namespace facesr
