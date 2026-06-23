#pragma once
/**
 * @file frosted_widget.h
 * @brief 毛玻璃面板组件
 *
 * 毛玻璃面板用于提升 GUI 演示观感。
 * 它通过截取父控件背景并做 separable blur 实现，不参与训练、推理或指标计算。
 */

#include "gui/style_config.h"
#include <QWidget>
#include <QPixmap>

namespace facesr {
namespace gui {

/**
 * @brief 毛玻璃面板组件
 * 使用栈模糊算法实现高性能毛玻璃效果
 */
class FrostedWidget : public QWidget {
public:
    explicit FrostedWidget(QWidget* parent = nullptr);

    // 设置毛玻璃参数
    void setBlurRadius(int radius);
    void setTintColor(const QColor& color);
    void setOpacity(int opacity);
    void setFrostedConfig(const FrostedConfig& config);

    // 启用/禁用毛玻璃效果
    void setEnabled(bool enabled);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    // 捕获背景
    QPixmap captureBackground();

    // 应用栈模糊算法
    QImage applyBlur(const QImage& source, int radius);

    // 栈模糊核心函数
    void stackBlurHorizontal(QImage& image, int radius);
    void stackBlurVertical(QImage& image, int radius);

    int blurRadius_ = 25;
    QColor tintColor_ = QColor("#FFFFFF");
    int opacity_ = 200;
    bool enabled_ = true;
};

}  // namespace gui
}  // namespace facesr
