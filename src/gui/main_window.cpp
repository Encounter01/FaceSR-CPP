/**
 * @file main_window.cpp
 * @brief Qt主窗口实现
 */

// 先包含 Qt 头文件
#include "gui/main_window.h"
#include "gui/style_config.h"
#include "gui/background_widget.h"
#include "gui/frosted_widget.h"

// 包含完整的 cv::Mat 定义
#include <opencv2/opencv.hpp>

// 避免 Qt 宏与 LibTorch 冲突
#ifdef slots
#undef slots
#endif
#ifdef signals
#undef signals
#endif

#include "inference.h"
#include "common/config.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QSplitter>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QStatusBar>
#include <QApplication>
#include <QResizeEvent>
#include <QDir>
#include <filesystem>

namespace facesr {
namespace gui {

// ==================== ProcessThread ====================

// PIMPL 实现
struct ProcessThread::Impl {
    Inference* inferencer;
    cv::Mat inputMat;

    Impl(Inference* inf, const cv::Mat& mat)
        : inferencer(inf), inputMat(mat) {}
};

// QPixmap 转 cv::Mat 辅助函数 (返回BGR格式, 与OpenCV/推理管道一致)
static cv::Mat pixmapToMat(const QPixmap& pixmap) {
    QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGB888);
    cv::Mat mat(image.height(), image.width(), CV_8UC3,
                const_cast<uchar*>(image.bits()), image.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

// cv::Mat 转 QPixmap 辅助函数
static QPixmap matToPixmap(const cv::Mat& mat) {
    cv::Mat rgb;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = mat;
    }
    QImage image(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(image.copy());
}

ProcessThread::ProcessThread(Inference* inferencer, const QPixmap& inputPixmap)
    : pImpl_(std::make_unique<Impl>(inferencer, pixmapToMat(inputPixmap))) {
}

ProcessThread::~ProcessThread() = default;

void ProcessThread::run() {
    try {
        emit progress(50);
        cv::Mat result = pImpl_->inferencer->process(pImpl_->inputMat);
        emit progress(100);
        emit finished(matToPixmap(result));
    } catch (const std::exception& e) {
        emit error(QString::fromStdString(e.what()));
    }
}


// ==================== BatchThread ====================

struct BatchThread::Impl {
    Inference* inferencer;
    std::string inputDir;
    std::string outputDir;

    Impl(Inference* inf, const std::string& inDir, const std::string& outDir)
        : inferencer(inf), inputDir(inDir), outputDir(outDir) {}
};

BatchThread::BatchThread(Inference* inferencer, const QString& inputDir, const QString& outputDir)
    : pImpl_(std::make_unique<Impl>(inferencer, inputDir.toStdString(), outputDir.toStdString())) {
}

BatchThread::~BatchThread() = default;

void BatchThread::run() {
    try {
        pImpl_->inferencer->process_folder(pImpl_->inputDir, pImpl_->outputDir);
        // 统计处理的文件数 (简单计算输出目录中的文件)
        int count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(pImpl_->outputDir)) {
            if (entry.is_regular_file()) count++;
        }
        emit finished(count);
    } catch (const std::exception& e) {
        emit error(QString::fromStdString(e.what()));
    }
}


// ==================== ImageLabel ====================

ImageLabel::ImageLabel(const QString& text, QWidget* parent)
    : QLabel(text, parent) {
    setAlignment(Qt::AlignCenter);
    setMinimumSize(300, 300);
    setStyleSheet(
        "QLabel {"
        "  background: rgba(255, 255, 255, 120);"
        "  border: 2px solid rgba(255, 255, 255, 180);"
        "  border-radius: 10px;"
        "}"
    );
}

void ImageLabel::setImage(const QPixmap& pixmap) {
    pixmap_ = pixmap;
    updateDisplay();
}

void ImageLabel::clearImage() {
    pixmap_ = QPixmap();
    clear();
    setText("无图像");
}

void ImageLabel::resizeEvent(QResizeEvent* event) {
    QLabel::resizeEvent(event);
    updateDisplay();
}

void ImageLabel::updateDisplay() {
    if (pixmap_.isNull()) return;

    QPixmap scaled = pixmap_.scaled(
        size(),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    );
    QLabel::setPixmap(scaled);
}


// ==================== MainWindow ====================

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    // 必须在 setupUI 之前加载配置
    loadUIConfig();

    setupUI();
    setupMenuBar();

    setWindowTitle("超分辨率人脸图像重建系统");
    setMinimumSize(1000, 700);
}

MainWindow::~MainWindow() {
    if (processThread_ && processThread_->isRunning()) {
        processThread_->wait();
    }
}

void MainWindow::setupUI() {
    // 1. 创建背景层（最底层）
    if (styleConfig_ && styleConfig_->enableCustomBackground()) {
        backgroundWidget_ = new BackgroundWidget(this);
        backgroundWidget_->setBackground(styleConfig_->background());
        backgroundWidget_->setGeometry(rect());
        backgroundWidget_->lower();
    }

    // 2. 创建透明中央控件
    QWidget* central = new QWidget(this);
    if (backgroundWidget_) {
        central->setAttribute(Qt::WA_TranslucentBackground);
    }
    setCentralWidget(central);

    QVBoxLayout* mainLayout = new QVBoxLayout(central);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(10, 10, 10, 10);

    // 3. 创建毛玻璃工具栏面板
    QWidget* toolbar = nullptr;
    if (styleConfig_ && styleConfig_->enableFrosted() && styleConfig_->frosted().enabled) {
        toolbarPanel_ = new FrostedWidget(central);
        toolbarPanel_->setFrostedConfig(styleConfig_->frosted());
        toolbar = toolbarPanel_;
    } else {
        toolbar = new QWidget(central);
    }

    QHBoxLayout* toolbarLayout = new QHBoxLayout(toolbar);
    toolbarLayout->setSpacing(10);
    toolbarLayout->setContentsMargins(15, 10, 15, 10);

    btnOpen_ = new QPushButton("打开图像");
    btnOpen_->setMinimumWidth(100);
    connect(btnOpen_, &QPushButton::clicked, this, &MainWindow::openImage);
    toolbarLayout->addWidget(btnOpen_);

    btnLoadModel_ = new QPushButton("加载模型");
    btnLoadModel_->setMinimumWidth(100);
    connect(btnLoadModel_, &QPushButton::clicked, this, &MainWindow::loadModel);
    toolbarLayout->addWidget(btnLoadModel_);

    btnProcess_ = new QPushButton("开始重建");
    btnProcess_->setMinimumWidth(100);
    btnProcess_->setEnabled(false);

    // 应用主题颜色
    QString primaryColor = "#4CAF50";
    QString accentColor = "#45a049";
    if (styleConfig_) {
        primaryColor = styleConfig_->theme().primaryColor.name();
        accentColor = styleConfig_->theme().accentColor.name();
    }

    btnProcess_->setStyleSheet(
        QString("QPushButton { "
                "  background-color: %1; "
                "  color: white; "
                "  font-weight: bold; "
                "  border: none; "
                "  border-radius: 8px; "
                "  padding: 10px 20px; "
                "} "
                "QPushButton:disabled { "
                "  background-color: #cccccc; "
                "} "
                "QPushButton:hover:enabled { "
                "  background-color: %2; "
                "}")
            .arg(primaryColor)
            .arg(accentColor)
    );
    connect(btnProcess_, &QPushButton::clicked, this, &MainWindow::processImage);
    toolbarLayout->addWidget(btnProcess_);

    btnSave_ = new QPushButton("保存结果");
    btnSave_->setMinimumWidth(100);
    btnSave_->setEnabled(false);
    connect(btnSave_, &QPushButton::clicked, this, &MainWindow::saveImage);
    toolbarLayout->addWidget(btnSave_);

    btnBatch_ = new QPushButton("批量处理");
    btnBatch_->setMinimumWidth(100);
    btnBatch_->setEnabled(false);
    connect(btnBatch_, &QPushButton::clicked, this, &MainWindow::batchProcess);
    toolbarLayout->addWidget(btnBatch_);

    toolbarLayout->addStretch();

    modelStatus_ = new QLabel("模型: 未加载");
    modelStatus_->setStyleSheet("color: #666;");
    toolbarLayout->addWidget(modelStatus_);

    mainLayout->addWidget(toolbar);

    // 图像显示区域
    QSplitter* splitter = new QSplitter(Qt::Horizontal);

    // 输入图像
    QGroupBox* inputGroup = new QGroupBox("输入图像 (低分辨率)");
    QVBoxLayout* inputLayout = new QVBoxLayout(inputGroup);
    inputLabel_ = new ImageLabel("点击\"打开图像\"选择图像");
    inputLayout->addWidget(inputLabel_);
    inputInfo_ = new QLabel("尺寸: -");
    inputLayout->addWidget(inputInfo_);
    splitter->addWidget(inputGroup);

    // 输出图像
    QGroupBox* outputGroup = new QGroupBox("输出图像 (高分辨率)");
    QVBoxLayout* outputLayout = new QVBoxLayout(outputGroup);
    outputLabel_ = new ImageLabel("处理后的图像将显示在这里");
    outputLayout->addWidget(outputLabel_);
    outputInfo_ = new QLabel("尺寸: -");
    outputLayout->addWidget(outputInfo_);
    splitter->addWidget(outputGroup);

    mainLayout->addWidget(splitter, 1);

    // 进度条
    progressBar_ = new QProgressBar();
    progressBar_->setVisible(false);
    mainLayout->addWidget(progressBar_);

    // 状态栏
    statusBar()->showMessage("就绪");
}

void MainWindow::setupMenuBar() {
    // 文件菜单
    QMenu* fileMenu = menuBar()->addMenu("文件");

    QAction* openAction = new QAction("打开图像", this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::openImage);
    fileMenu->addAction(openAction);

    QAction* saveAction = new QAction("保存结果", this);
    saveAction->setShortcut(QKeySequence::Save);
    connect(saveAction, &QAction::triggered, this, &MainWindow::saveImage);
    fileMenu->addAction(saveAction);

    fileMenu->addSeparator();

    QAction* exitAction = new QAction("退出", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QMainWindow::close);
    fileMenu->addAction(exitAction);

    // 模型菜单
    QMenu* modelMenu = menuBar()->addMenu("模型");
    QAction* loadModelAction = new QAction("加载模型", this);
    connect(loadModelAction, &QAction::triggered, this, &MainWindow::loadModel);
    modelMenu->addAction(loadModelAction);

    // 帮助菜单
    QMenu* helpMenu = menuBar()->addMenu("帮助");
    QAction* aboutAction = new QAction("关于", this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::showAbout);
    helpMenu->addAction(aboutAction);
}

void MainWindow::openImage() {
    QString filePath = QFileDialog::getOpenFileName(
        this,
        "选择图像",
        "",
        "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"
    );

    if (!filePath.isEmpty()) {
        inputPixmap_ = QPixmap(filePath);

        if (inputPixmap_.isNull()) {
            QMessageBox::critical(this, "错误", "无法加载图像");
            return;
        }

        currentImagePath_ = filePath;
        inputLabel_->setImage(inputPixmap_);
        inputInfo_->setText(QString("尺寸: %1 x %2")
            .arg(inputPixmap_.width())
            .arg(inputPixmap_.height()));

        outputLabel_->clearImage();
        outputPixmap_ = QPixmap();
        outputInfo_->setText("尺寸: -");

        updateButtonStates();
        updateStatus("已加载图像: " + filePath);
    }
}

void MainWindow::loadModel() {
    // 搜索checkpoints目录: 先尝试exe目录, 再向上搜索
    QString startDir;
    {
        QDir dir(QApplication::applicationDirPath());
        if (QDir(dir.filePath("checkpoints")).exists()) {
            startDir = dir.filePath("checkpoints");
        } else {
            for (int i = 0; i < 5; ++i) {
                if (!dir.cdUp()) break;
                if (QDir(dir.filePath("checkpoints")).exists()) {
                    startDir = dir.filePath("checkpoints");
                    break;
                }
            }
        }
    }

    QString filePath = QFileDialog::getOpenFileName(
        this,
        "选择模型文件",
        startDir,
        "PyTorch模型 (*.pt *.pth);;所有文件 (*.*)"
    );

    if (!filePath.isEmpty()) {
        updateStatus("正在加载模型...");
        QApplication::processEvents();

        try {
            // 使用Auto模式: 有GPU时自动启用GPU+CPU混合模式
            inferencer_ = std::make_unique<Inference>(
                filePath.toStdString(), 4, DeviceType::Auto);

            if (inferencer_->is_loaded()) {
                QFileInfo fileInfo(filePath);
                modelStatus_->setText("模型: " + fileInfo.fileName());
                modelStatus_->setStyleSheet("color: #4CAF50; font-weight: bold;");

                updateButtonStates();
                updateStatus("模型加载成功: " + filePath);
            } else {
                QMessageBox::critical(this, "错误", "模型加载失败");
                updateStatus("模型加载失败");
            }
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "错误",
                QString("模型加载失败: %1").arg(e.what()));
            updateStatus("模型加载失败");
        }
    }
}

void MainWindow::processImage() {
    if (inputPixmap_.isNull() || !inferencer_) return;

    btnProcess_->setEnabled(false);
    progressBar_->setVisible(true);
    progressBar_->setValue(0);
    updateStatus("正在处理...");

    processThread_ = new ProcessThread(inferencer_.get(), inputPixmap_);
    connect(processThread_, &ProcessThread::finished,
            this, &MainWindow::onProcessFinished);
    connect(processThread_, &ProcessThread::error,
            this, &MainWindow::onProcessError);
    connect(processThread_, &ProcessThread::progress,
            progressBar_, &QProgressBar::setValue);
    connect(processThread_, &ProcessThread::finished,
            processThread_, &QThread::deleteLater);
    processThread_->start();
}

void MainWindow::onProcessFinished(QPixmap result) {
    outputPixmap_ = result;
    outputLabel_->setImage(outputPixmap_);
    outputInfo_->setText(QString("尺寸: %1 x %2")
        .arg(outputPixmap_.width())
        .arg(outputPixmap_.height()));

    progressBar_->setVisible(false);
    updateButtonStates();
    updateStatus("处理完成");
}

void MainWindow::onProcessError(QString message) {
    progressBar_->setVisible(false);
    updateButtonStates();
    QMessageBox::critical(this, "错误", "处理失败: " + message);
    updateStatus("处理失败");
}

void MainWindow::saveImage() {
    if (outputPixmap_.isNull()) {
        QMessageBox::warning(this, "警告", "没有可保存的结果");
        return;
    }

    QString filePath = QFileDialog::getSaveFileName(
        this,
        "保存图像",
        "result_sr.png",
        "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*.*)"
    );

    if (!filePath.isEmpty()) {
        if (outputPixmap_.save(filePath)) {
            updateStatus("图像已保存: " + filePath);
            QMessageBox::information(this, "成功",
                "图像已保存到:\n" + filePath);
        } else {
            QMessageBox::critical(this, "错误", "保存失败");
        }
    }
}

void MainWindow::batchProcess() {
    if (!inferencer_) {
        QMessageBox::warning(this, "警告", "请先加载模型");
        return;
    }

    QString inputDir = QFileDialog::getExistingDirectory(
        this, "选择输入文件夹", "");
    if (inputDir.isEmpty()) return;

    QString outputDir = QFileDialog::getExistingDirectory(
        this, "选择输出文件夹", "");
    if (outputDir.isEmpty()) return;

    // 禁用按钮, 在后台线程中执行批量处理
    btnBatch_->setEnabled(false);
    btnProcess_->setEnabled(false);
    progressBar_->setVisible(true);
    progressBar_->setRange(0, 0);  // 不确定进度, 显示忙碌动画
    updateStatus("正在批量处理...");

    auto* batchThread = new BatchThread(inferencer_.get(), inputDir, outputDir);
    connect(batchThread, &BatchThread::finished,
            this, &MainWindow::onBatchFinished);
    connect(batchThread, &BatchThread::error,
            this, &MainWindow::onBatchError);
    connect(batchThread, &BatchThread::finished,
            batchThread, &QThread::deleteLater);
    connect(batchThread, &BatchThread::error,
            batchThread, &QThread::deleteLater);
    batchThread->start();
}

void MainWindow::onBatchFinished(int count) {
    progressBar_->setVisible(false);
    progressBar_->setRange(0, 100);
    updateButtonStates();
    updateStatus(QString("批量处理完成, 共处理 %1 张图像").arg(count));
    QMessageBox::information(this, "成功",
        QString("批量处理完成!\n共处理 %1 张图像").arg(count));
}

void MainWindow::onBatchError(QString message) {
    progressBar_->setVisible(false);
    progressBar_->setRange(0, 100);
    updateButtonStates();
    QMessageBox::critical(this, "错误", "批量处理失败: " + message);
    updateStatus("批量处理失败");
}

void MainWindow::updateButtonStates() {
    bool hasModel = inferencer_ && inferencer_->is_loaded();
    bool hasInput = !inputPixmap_.isNull();
    bool hasOutput = !outputPixmap_.isNull();

    btnProcess_->setEnabled(hasModel && hasInput);
    btnSave_->setEnabled(hasOutput);
    btnBatch_->setEnabled(hasModel);
}

void MainWindow::updateStatus(const QString& message) {
    statusBar()->showMessage(message);
}

void MainWindow::showAbout() {
    QMessageBox::about(this, "关于",
        "<h2>超分辨率人脸图像重建系统</h2>"
        "<p>版本: 1.1.0 (C++)</p>"
        "<p>基于LibTorch和Qt开发</p>"
        "<p>功能特点:</p>"
        "<ul>"
        "<li>4倍超分辨率放大</li>"
        "<li>RRDB生成器网络</li>"
        "<li>GPU+CPU混合流水线加速</li>"
        "<li>三阶段并行: CPU预处理 → GPU推理 → CPU后处理</li>"
        "<li>批量处理功能</li>"
        "</ul>"
        "<p>作者: Encounter01</p>"
    );
}

void MainWindow::loadUIConfig() {
    styleConfig_ = &StyleConfig::instance();
    styleConfig_->loadFromFile("config/ui_config.ini");
}

void MainWindow::setupBackground() {
    if (!styleConfig_ || !backgroundWidget_) {
        return;
    }

    backgroundWidget_->setBackground(styleConfig_->background());
}

void MainWindow::applyTheme() {
    if (!styleConfig_) {
        return;
    }

    // 应用主题到整个窗口（如果需要的话可以在这里添加全局样式）
    // 目前主题色主要应用在按钮上，已在 setupUI 中处理
}

void MainWindow::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);

    // 同步背景尺寸
    if (backgroundWidget_) {
        backgroundWidget_->setGeometry(rect());
    }
}

}  // namespace gui
}  // namespace facesr
