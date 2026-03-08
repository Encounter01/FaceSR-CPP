/**
 * @file frosted_widget.cpp
 * @brief 毛玻璃面板组件实现
 */

#include "gui/frosted_widget.h"
#include <QPainter>
#include <QPainterPath>
#include <QImage>
#include <QtMath>
#include <algorithm>

namespace facesr {
namespace gui {

FrostedWidget::FrostedWidget(QWidget* parent)
    : QWidget(parent) {
    setAttribute(Qt::WA_TranslucentBackground);
}

void FrostedWidget::setBlurRadius(int radius) {
    blurRadius_ = qBound(0, radius, 50);
    update();
}

void FrostedWidget::setTintColor(const QColor& color) {
    tintColor_ = color;
    update();
}

void FrostedWidget::setOpacity(int opacity) {
    opacity_ = qBound(0, opacity, 255);
    update();
}

void FrostedWidget::setFrostedConfig(const FrostedConfig& config) {
    enabled_ = config.enabled;
    blurRadius_ = qBound(0, config.blurRadius, 50);
    tintColor_ = config.tintColor;
    opacity_ = qBound(0, config.opacity, 255);
    update();
}

void FrostedWidget::setEnabled(bool enabled) {
    enabled_ = enabled;
    update();
}

void FrostedWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    if (!enabled_ || blurRadius_ == 0) {
        // 禁用毛玻璃或无模糊，绘制简单半透明面板
        QColor bgColor = tintColor_;
        bgColor.setAlpha(opacity_);

        QPainterPath path;
        path.addRoundedRect(rect(), 10, 10);
        painter.fillPath(path, bgColor);
        return;
    }

    // 捕获背景
    QPixmap background = captureBackground();
    if (background.isNull()) {
        // 无法捕获背景，使用简单半透明面板
        QColor bgColor = tintColor_;
        bgColor.setAlpha(opacity_);
        QPainterPath path;
        path.addRoundedRect(rect(), 10, 10);
        painter.fillPath(path, bgColor);
        return;
    }

    // 应用模糊
    QImage blurred = applyBlur(background.toImage(), blurRadius_);

    // 创建圆角路径
    QPainterPath path;
    path.addRoundedRect(rect(), 10, 10);
    painter.setClipPath(path);

    // 绘制模糊背景
    painter.drawImage(0, 0, blurred);

    // 叠加半透明色调层
    QColor tint = tintColor_;
    tint.setAlpha(opacity_);
    painter.fillPath(path, tint);

    // 绘制边框
    painter.setClipping(false);
    QPen pen(QColor(255, 255, 255, 100), 2);
    painter.setPen(pen);
    painter.drawRoundedRect(rect().adjusted(1, 1, -1, -1), 10, 10);
}

QPixmap FrostedWidget::captureBackground() {
    if (!parentWidget()) {
        return QPixmap();
    }

    // 捕获父控件在当前位置的内容
    QPixmap pixmap(size());
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.translate(-pos());

    // 渲染父控件的背景
    parentWidget()->render(&painter, QPoint(), QRegion(),
        QWidget::DrawWindowBackground | QWidget::DrawChildren);

    return pixmap;
}

QImage FrostedWidget::applyBlur(const QImage& source, int radius) {
    if (radius <= 0) {
        return source;
    }

    // 转换为 ARGB32 格式
    QImage result = source.convertToFormat(QImage::Format_ARGB32);

    // 应用栈模糊（水平 + 垂直）
    stackBlurHorizontal(result, radius);
    stackBlurVertical(result, radius);

    return result;
}

void FrostedWidget::stackBlurHorizontal(QImage& image, int radius) {
    int width = image.width();
    int height = image.height();

    if (width <= 0 || height <= 0 || radius <= 0) {
        return;
    }

    int div = 2 * radius + 1;

    for (int y = 0; y < height; ++y) {
        QRgb* line = reinterpret_cast<QRgb*>(image.scanLine(y));

        // 复制原始行数据, 避免滑动窗口读取已修改的像素
        std::vector<QRgb> original(line, line + width);

        int sumR = 0, sumG = 0, sumB = 0, sumA = 0;

        // 初始化：累加左侧窗口
        for (int x = -radius; x <= radius; ++x) {
            int px = qBound(0, x, width - 1);
            QRgb pixel = original[px];
            sumR += qRed(pixel);
            sumG += qGreen(pixel);
            sumB += qBlue(pixel);
            sumA += qAlpha(pixel);
        }

        // 滑动窗口
        for (int x = 0; x < width; ++x) {
            line[x] = qRgba(sumR / div, sumG / div, sumB / div, sumA / div);

            // 移除左边的像素 (从原始数据读取)
            int leftX = qBound(0, x - radius, width - 1);
            QRgb leftPixel = original[leftX];
            sumR -= qRed(leftPixel);
            sumG -= qGreen(leftPixel);
            sumB -= qBlue(leftPixel);
            sumA -= qAlpha(leftPixel);

            // 添加右边的像素 (从原始数据读取)
            int rightX = qBound(0, x + radius + 1, width - 1);
            QRgb rightPixel = original[rightX];
            sumR += qRed(rightPixel);
            sumG += qGreen(rightPixel);
            sumB += qBlue(rightPixel);
            sumA += qAlpha(rightPixel);
        }
    }
}

void FrostedWidget::stackBlurVertical(QImage& image, int radius) {
    int width = image.width();
    int height = image.height();

    if (width <= 0 || height <= 0 || radius <= 0) {
        return;
    }

    int div = 2 * radius + 1;

    for (int x = 0; x < width; ++x) {
        // 复制原始列数据, 避免滑动窗口读取已修改的像素
        std::vector<QRgb> original(height);
        for (int y = 0; y < height; ++y) {
            original[y] = image.pixel(x, y);
        }

        int sumR = 0, sumG = 0, sumB = 0, sumA = 0;

        // 初始化：累加上方窗口
        for (int y = -radius; y <= radius; ++y) {
            int py = qBound(0, y, height - 1);
            QRgb pixel = original[py];
            sumR += qRed(pixel);
            sumG += qGreen(pixel);
            sumB += qBlue(pixel);
            sumA += qAlpha(pixel);
        }

        // 滑动窗口
        for (int y = 0; y < height; ++y) {
            image.setPixel(x, y, qRgba(sumR / div, sumG / div, sumB / div, sumA / div));

            // 移除上方的像素 (从原始数据读取)
            int topY = qBound(0, y - radius, height - 1);
            QRgb topPixel = original[topY];
            sumR -= qRed(topPixel);
            sumG -= qGreen(topPixel);
            sumB -= qBlue(topPixel);
            sumA -= qAlpha(topPixel);

            // 添加下方的像素 (从原始数据读取)
            int bottomY = qBound(0, y + radius + 1, height - 1);
            QRgb bottomPixel = original[bottomY];
            sumR += qRed(bottomPixel);
            sumG += qGreen(bottomPixel);
            sumB += qBlue(bottomPixel);
            sumA += qAlpha(bottomPixel);
        }
    }
}

}  // namespace gui
}  // namespace facesr
