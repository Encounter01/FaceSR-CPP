#pragma once
/**
 * @file style_config.h
 * @brief UI样式配置管理（单例模式）
 *
 * 样式配置只影响演示界面外观，不影响模型推理结果。
 * 独立成单例是为了让背景、毛玻璃面板和主窗口能共享同一份 UI 配置。
 */

#include <QString>
#include <QColor>

namespace facesr {
namespace gui {

/**
 * @brief 背景类型枚举
 */
enum class BackgroundType {
    Solid,      // 纯色
    Gradient,   // 渐变
    Image       // 图片
};

/**
 * @brief 渐变类型枚举
 */
enum class GradientType {
    Linear,     // 线性渐变
    Radial      // 径向渐变
};

/**
 * @brief 背景配置结构体
 */
struct BackgroundConfig {
    BackgroundType type = BackgroundType::Gradient;
    GradientType gradientType = GradientType::Linear;

    // 纯色配置
    QColor solidColor = QColor("#2C3E50");

    // 渐变配置
    QColor startColor = QColor("#2C3E50");
    QColor endColor = QColor("#3498DB");
    qreal startX = 0.0;
    qreal startY = 0.0;
    qreal endX = 1.0;
    qreal endY = 1.0;
    qreal centerX = 0.5;  // 径向渐变中心
    qreal centerY = 0.5;
    qreal radius = 1.0;   // 径向渐变半径

    // 图片配置
    QString imagePath;
};

/**
 * @brief 毛玻璃配置结构体
 */
struct FrostedConfig {
    bool enabled = true;
    int blurRadius = 25;           // 模糊半径 (0-50)
    int opacity = 200;             // 不透明度 (0-255)
    QColor tintColor = QColor("#FFFFFF");  // 色调颜色
};

/**
 * @brief 主题配置结构体
 */
struct ThemeConfig {
    QColor primaryColor = QColor("#4CAF50");
    QColor accentColor = QColor("#45a049");
    QColor textColor = QColor("#FFFFFF");
    QColor borderColor = QColor(255, 255, 255, 100);
    QColor panelBackground = QColor(255, 255, 255, 150);
};

/**
 * @brief 样式配置管理类（单例模式）
 */
class StyleConfig {
public:
    // 获取单例实例
    static StyleConfig& instance();

    // 禁用拷贝和赋值
    StyleConfig(const StyleConfig&) = delete;
    StyleConfig& operator=(const StyleConfig&) = delete;

    // 加载和保存配置
    bool loadFromFile(const QString& filePath);
    bool saveToFile(const QString& filePath);

    // 配置访问
    const BackgroundConfig& background() const { return background_; }
    const FrostedConfig& frosted() const { return frosted_; }
    const ThemeConfig& theme() const { return theme_; }

    bool enableFrosted() const { return enableFrosted_; }
    bool enableCustomBackground() const { return enableCustomBackground_; }

    // 配置修改
    void setBackground(const BackgroundConfig& config) { background_ = config; }
    void setFrosted(const FrostedConfig& config) { frosted_ = config; }
    void setTheme(const ThemeConfig& config) { theme_ = config; }
    void setEnableFrosted(bool enable) { enableFrosted_ = enable; }
    void setEnableCustomBackground(bool enable) { enableCustomBackground_ = enable; }

private:
    StyleConfig();  // 私有构造函数
    ~StyleConfig() = default;

    // 加载默认配置
    void loadDefaults();

    // 辅助函数：解析颜色字符串
    static QColor parseColorString(const QString& str);

    // 配置数据
    bool enableFrosted_ = true;
    bool enableCustomBackground_ = true;
    BackgroundConfig background_;
    FrostedConfig frosted_;
    ThemeConfig theme_;
};

}  // namespace gui
}  // namespace facesr
