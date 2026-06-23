/**
 * @file main_window.cpp
 * @brief Qt主窗口实现
 *
 * 读 GUI 代码时可以按用户操作顺序看：
 * openImage() -> loadModel() -> processImage() -> onProcessFinished() -> saveImage()。
 * 其中 processImage() 只负责启动线程，真正的算法推理仍由 Inference::process() 完成。
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
#include <QFileInfo>
#include <QInputDialog>
#include <QSet>
#include <QSizePolicy>
#include <filesystem>
#include <vector>

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

// QPixmap 转 cv::Mat 辅助函数。
// Qt 显示使用 RGB，OpenCV/Inference 入口约定 BGR，因此这里显式转换，避免颜色通道错位。
// 返回 BGR 格式，与 OpenCV/推理管道一致。
static cv::Mat pixmapToMat(const QPixmap& pixmap) {
    QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGB888);
    cv::Mat mat(image.height(), image.width(), CV_8UC3,
                const_cast<uchar*>(image.bits()), image.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

// cv::Mat 转 QPixmap 辅助函数。
// 推理结果从 OpenCV BGR 回到 Qt RGB 显示。
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
        // 这里运行在后台线程中，避免大图推理或 GPU 同步时阻塞 UI 事件循环。
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
    std::vector<std::string> inputFiles;
    std::string outputDir;

    Impl(Inference* inf, const QStringList& files, const std::string& outDir)
        : inferencer(inf), outputDir(outDir) {
        inputFiles.reserve(files.size());
        for (const QString& file : files) {
            inputFiles.push_back(file.toStdString());
        }
    }
};

BatchThread::BatchThread(Inference* inferencer, const QStringList& inputFiles, const QString& outputDir)
    : pImpl_(std::make_unique<Impl>(inferencer, inputFiles, outputDir.toStdString())) {
}

BatchThread::~BatchThread() = default;

void BatchThread::run() {
    try {
        std::filesystem::create_directories(pImpl_->outputDir);
        int count = 0;
        for (const std::string& inputPath : pImpl_->inputFiles) {
            std::filesystem::path input(inputPath);
            std::string ext = input.extension().string();
            std::string outputPath =
                (std::filesystem::path(pImpl_->outputDir) /
                 (input.stem().string() + "_sr" + ext)).string();

            if (pImpl_->inferencer->process_file(inputPath, outputPath)) {
                ++count;
            }
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

void ImageLabel::setMaxDisplaySize(const QSize& maxSize) {
    maxDisplaySize_ = maxSize;
    updateDisplay();
}

void ImageLabel::resizeEvent(QResizeEvent* event) {
    QLabel::resizeEvent(event);
    updateDisplay();
}

void ImageLabel::updateDisplay() {
    if (pixmap_.isNull()) return;

    QSize targetSize = size();
    if (maxDisplaySize_.isValid()) {
        targetSize = targetSize.boundedTo(maxDisplaySize_);
    }

    QPixmap scaled = pixmap_.scaled(
        targetSize,
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    );
    QLabel::setPixmap(scaled);
}


// ==================== MainWindow ====================

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    // 蹇呴』鍦?setupUI 涔嬪墠鍔犺浇閰嶇疆
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
    inputGroup->setMinimumWidth(240);
    QVBoxLayout* inputLayout = new QVBoxLayout(inputGroup);
    inputLabel_ = new ImageLabel("点击\"打开图像\"选择图像");
    inputLabel_->setMinimumSize(180, 180);
    inputLabel_->setMaxDisplaySize(QSize(180, 180));
    inputLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    inputLayout->addWidget(inputLabel_);
    inputInfo_ = new QLabel("尺寸: -");
    inputLayout->addWidget(inputInfo_);
    splitter->addWidget(inputGroup);

    // 输出图像
    QGroupBox* outputGroup = new QGroupBox("输出图像 (高分辨率)");
    outputGroup->setMinimumWidth(520);
    QVBoxLayout* outputLayout = new QVBoxLayout(outputGroup);
    outputLabel_ = new ImageLabel("处理后的图像将显示在这里");
    outputLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    outputLayout->addWidget(outputLabel_);
    outputInfo_ = new QLabel("尺寸: -");
    outputLayout->addWidget(outputInfo_);
    splitter->addWidget(outputGroup);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 3);
    splitter->setSizes({280, 720});

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
    // 搜索 checkpoints 目录：优先 checkpoints/final，不存在则回退到 checkpoints。
    // 先在 exe 目录尝试，再向上搜索。
    QString startDir;
    {
        QDir dir(QApplication::applicationDirPath());
        auto findCheckpointDir = [&dir]() -> QString {
            QDir finalDir(dir.filePath("checkpoints/final"));
            if (finalDir.exists()) return finalDir.absolutePath();
            QDir ckptDir(dir.filePath("checkpoints"));
            if (ckptDir.exists()) return ckptDir.absolutePath();
            return QString();
        };

        startDir = findCheckpointDir();
        if (startDir.isEmpty()) {
            for (int i = 0; i < 5; ++i) {
                if (!dir.cdUp()) break;
                startDir = findCheckpointDir();
                if (!startDir.isEmpty()) break;
            }
        }
    }

    QString filePath = QFileDialog::getOpenFileName(
        this,
        "选择模型文件",
        startDir,
        "生成器模型 (generator_*.pt facesr_*.pt *.pth);;所有 PyTorch 文件 (*.pt *.pth);;所有文件 (*.*)"
    );

    if (!filePath.isEmpty()) {
        updateStatus("正在加载模型...");
        QApplication::processEvents();

        try {
            // 使用 Auto 模式：有 CUDA 时使用 GPU，否则回退 CPU。
            // GUI 目前没有 attention 开关，因此适合加载默认 A4 结构模型。
            // 若权重来自 A5/CBAM，建议使用命令行 --attention 推理。
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
                static const QString kLoadHint =
                    "模型加载失败。可能原因：\n\n"
                    "1) 文件是判别器（discriminator_*.pt），GUI 只支持生成器。\n"
                    "2) 文件是 A5/CBAM 注意力模型，GUI 默认不启用 attention。\n"
                    "   请改用命令行：facesr_test --attention --model <文件>\n"
                    "3) 文件结构与本项目 RRDBNet 不一致，或文件已损坏。\n\n"
                    "推荐使用：checkpoints/final/facesr_a4_best_psnr28.6019.pt";
                QMessageBox::critical(this, "模型加载失败", kLoadHint);
                updateStatus("模型加载失败");
            }
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "模型加载失败",
                QString("加载过程抛出异常：%1\n\n"
                        "建议改用 checkpoints/final/facesr_a4_best_psnr28.6019.pt").arg(e.what()));
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

    // 后台线程持有 inferencer_ 的裸指针，但 MainWindow 持有 unique_ptr 生命周期。
    // 处理期间按钮会被禁用，避免用户重复启动推理造成并发访问。
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

    const QStringList modes = {
        "选择图片文件",
        "选择图片文件夹"
    };
    bool ok = false;
    const QString mode = QInputDialog::getItem(
        this, "批量处理", "输入来源:", modes, 0, false, &ok);
    if (!ok || mode.isEmpty()) return;

    QStringList inputFiles;
    const QString imageFilter =
        "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*.*)";

    if (mode == modes.first()) {
        inputFiles = QFileDialog::getOpenFileNames(
            this, "选择输入图片", "", imageFilter);
        if (inputFiles.isEmpty()) return;
    } else {
        const QString inputDir = QFileDialog::getExistingDirectory(
            this, "选择输入文件夹", "");
        if (inputDir.isEmpty()) return;

        const QSet<QString> supportedExtensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff"
        };
        const QFileInfoList entries =
            QDir(inputDir).entryInfoList(QDir::Files | QDir::Readable, QDir::Name);
        for (const QFileInfo& entry : entries) {
            const QString ext = "." + entry.suffix().toLower();
            if (supportedExtensions.contains(ext)) {
                inputFiles.push_back(entry.absoluteFilePath());
            }
        }
    }

    if (inputFiles.isEmpty()) {
        QMessageBox::warning(this, "警告", "没有找到可处理的图片文件");
        return;
    }

    const QString outputDir = QFileDialog::getExistingDirectory(
        this, "选择输出文件夹", "");
    if (outputDir.isEmpty()) return;

    btnBatch_->setEnabled(false);
    btnProcess_->setEnabled(false);
    progressBar_->setVisible(true);
    progressBar_->setRange(0, 0);
    updateStatus("正在批量处理...");

    auto* batchThread = new BatchThread(inferencer_.get(), inputFiles, outputDir);
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
    updateStatus(QString("批量处理完成，共处理 %1 张图像").arg(count));
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
        "<p>基于 LibTorch 和 Qt 开发</p>"
        "<p>功能特点:</p>"
        "<ul>"
        "<li>4倍超分辨率放大</li>"
        "<li>RRDB生成器网络</li>"
        "<li>GPU+CPU混合流水线加速</li>"
        "<li>三阶段并行：CPU预处理 -> GPU推理 -> CPU后处理</li>"
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

    // 应用主题到整个窗口（如果需要的话可以在这里添加全局样式）。
    // 目前主题色主要应用在按钮上，已在 setupUI 中处理。
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
