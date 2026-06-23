/**
 * @file background_widget.cpp
 * @brief 背景渲染组件实现
 *
 * 背景绘制走缓存策略：paintEvent 只贴图，配置或尺寸变化时才重新 renderBackground()。
 */

#include "gui/background_widget.h"
#include <QPainter>
#include <QLinearGradient>
#include <QRadialGradient>
#include <QResizeEvent>

namespace facesr {
namespace gui {

BackgroundWidget::BackgroundWidget(QWidget* parent)
    : QWidget(parent) {
    setAttribute(Qt::WA_StyledBackground, true);
    setAutoFillBackground(true);
}

void BackgroundWidget::setBackground(const BackgroundConfig& config) {
    config_ = config;
    needsUpdate_ = true;
    update();
}

void BackgroundWidget::updateBackgroundCache() {
    needsUpdate_ = true;
    update();
}

void BackgroundWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    needsUpdate_ = true;
}

void BackgroundWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    if (needsUpdate_ || cachedBackground_.size() != size()) {
        // 只有配置变化或窗口尺寸变化时才重建缓存，避免频繁重绘造成 UI 卡顿。
        renderBackground();
        needsUpdate_ = false;
    }

    QPainter painter(this);
    painter.drawPixmap(0, 0, cachedBackground_);
}

void BackgroundWidget::renderBackground() {
    // 创建缓存 pixmap
    cachedBackground_ = QPixmap(size());
    QPainter painter(&cachedBackground_);
    painter.setRenderHint(QPainter::Antialiasing);

    switch (config_.type) {
        case BackgroundType::Solid: {
            // 纯色背景
            painter.fillRect(rect(), config_.solidColor);
            break;
        }

        case BackgroundType::Gradient: {
            QGradient* gradient = nullptr;

            if (config_.gradientType == GradientType::Linear) {
                // 线性渐变
                QLinearGradient* linearGrad = new QLinearGradient(
                    config_.startX * width(),
                    config_.startY * height(),
                    config_.endX * width(),
                    config_.endY * height()
                );
                linearGrad->setColorAt(0, config_.startColor);
                linearGrad->setColorAt(1, config_.endColor);
                gradient = linearGrad;
            } else {
                // 径向渐变
                QRadialGradient* radialGrad = new QRadialGradient(
                    config_.centerX * width(),
                    config_.centerY * height(),
                    config_.radius * qMax(width(), height()) / 2
                );
                radialGrad->setColorAt(0, config_.startColor);
                radialGrad->setColorAt(1, config_.endColor);
                gradient = radialGrad;
            }

            if (gradient) {
                painter.fillRect(rect(), *gradient);
                delete gradient;
            }
            break;
        }

        case BackgroundType::Image: {
            // 图片背景
            if (!config_.imagePath.isEmpty()) {
                QPixmap bgImage(config_.imagePath);
                if (!bgImage.isNull()) {
                    // 缩放图片以填充整个窗口（保持宽高比，可能裁剪）
                    QPixmap scaled = bgImage.scaled(
                        size(),
                        Qt::KeepAspectRatioByExpanding,
                        Qt::SmoothTransformation
                    );

                    // 居中绘制
                    int x = (width() - scaled.width()) / 2;
                    int y = (height() - scaled.height()) / 2;
                    painter.drawPixmap(x, y, scaled);
                } else {
                    // 图片加载失败，使用备用纯色
                    painter.fillRect(rect(), config_.solidColor);
                }
            } else {
                // 未设置图片路径，使用备用纯色
                painter.fillRect(rect(), config_.solidColor);
            }
            break;
        }
    }
}

}  // namespace gui
}  // namespace facesr
