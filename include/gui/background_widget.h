#pragma once
/**
 * @file background_widget.h
 * @brief 背景渲染组件
 *
 * 该组件只负责 GUI 背景绘制，与超分模型无关。
 * 背景内容会缓存到 QPixmap，窗口大小变化时再重新渲染，避免每次 paintEvent 都重复计算渐变或缩放图片。
 */

#include "gui/style_config.h"
#include <QWidget>
#include <QPixmap>

namespace facesr {
namespace gui {

/**
 * @brief 背景渲染组件
 * 支持纯色、渐变、图片三种背景类型
 */
class BackgroundWidget : public QWidget {
public:
    explicit BackgroundWidget(QWidget* parent = nullptr);

    // 设置背景配置
    void setBackground(const BackgroundConfig& config);

    // 更新背景缓存（窗口大小变化时调用）
    void updateBackgroundCache();

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    // 绘制背景到缓存
    void renderBackground();

    BackgroundConfig config_;
    QPixmap cachedBackground_;
    bool needsUpdate_ = true;
};

}  // namespace gui
}  // namespace facesr
