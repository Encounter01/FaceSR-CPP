/**
 * @file style_config.cpp
 * @brief UI样式配置管理实现
 */

#include "gui/style_config.h"
#include <QSettings>
#include <QFileInfo>
#include <QApplication>
#include <QDir>

namespace facesr {
namespace gui {

StyleConfig& StyleConfig::instance() {
    static StyleConfig instance;
    return instance;
}

StyleConfig::StyleConfig() {
    loadDefaults();
}

void StyleConfig::loadDefaults() {
    // 通用设置
    enableFrosted_ = true;
    enableCustomBackground_ = true;

    // 背景默认配置：蓝色渐变
    background_.type = BackgroundType::Gradient;
    background_.gradientType = GradientType::Linear;
    background_.solidColor = QColor("#2C3E50");
    background_.startColor = QColor("#2C3E50");
    background_.endColor = QColor("#3498DB");
    background_.startX = 0.0;
    background_.startY = 0.0;
    background_.endX = 1.0;
    background_.endY = 1.0;
    background_.centerX = 0.5;
    background_.centerY = 0.5;
    background_.radius = 1.0;
    background_.imagePath = "";

    // 毛玻璃默认配置
    frosted_.enabled = true;
    frosted_.blurRadius = 25;
    frosted_.opacity = 200;
    frosted_.tintColor = QColor("#FFFFFF");

    // 主题默认配置：绿色主题
    theme_.primaryColor = QColor("#4CAF50");
    theme_.accentColor = QColor("#45a049");
    theme_.textColor = QColor("#FFFFFF");
    theme_.borderColor = QColor(255, 255, 255, 100);
    theme_.panelBackground = QColor(255, 255, 255, 150);
}

bool StyleConfig::loadFromFile(const QString& filePath) {
    // 构建完整路径
    QString fullPath = filePath;
    if (QFileInfo(filePath).isRelative()) {
        fullPath = QApplication::applicationDirPath() + "/" + filePath;

        // 若exe目录下找不到, 向上搜索最多5级父目录 (解决从build子目录运行的问题)
        if (!QFileInfo::exists(fullPath)) {
            QDir dir(QApplication::applicationDirPath());
            for (int i = 0; i < 5; ++i) {
                if (!dir.cdUp()) break;
                QString candidate = dir.filePath(filePath);
                if (QFileInfo::exists(candidate)) {
                    fullPath = candidate;
                    break;
                }
            }
        }
    }

    // 检查文件是否存在
    if (!QFileInfo::exists(fullPath)) {
        qWarning("Config file not found: %s, using defaults", qPrintable(fullPath));
        return false;
    }

    QSettings settings(fullPath, QSettings::IniFormat);
    #if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    settings.setIniCodec("UTF-8");
    #endif

    // 读取通用设置
    settings.beginGroup("general");
    enableFrosted_ = settings.value("enable_frosted", true).toBool();
    enableCustomBackground_ = settings.value("enable_custom_background", true).toBool();
    settings.endGroup();

    // 读取背景配置
    settings.beginGroup("background");
    QString typeStr = settings.value("type", "gradient").toString().toLower();
    if (typeStr == "solid") {
        background_.type = BackgroundType::Solid;
    } else if (typeStr == "gradient") {
        background_.type = BackgroundType::Gradient;
    } else if (typeStr == "image") {
        background_.type = BackgroundType::Image;
    }

    QString gradientTypeStr = settings.value("gradient_type", "linear").toString().toLower();
    background_.gradientType = (gradientTypeStr == "radial") ?
        GradientType::Radial : GradientType::Linear;

    background_.solidColor = QColor(settings.value("solid_color", "#2C3E50").toString());
    background_.startColor = QColor(settings.value("start_color", "#2C3E50").toString());
    background_.endColor = QColor(settings.value("end_color", "#3498DB").toString());
    background_.startX = settings.value("start_x", 0.0).toDouble();
    background_.startY = settings.value("start_y", 0.0).toDouble();
    background_.endX = settings.value("end_x", 1.0).toDouble();
    background_.endY = settings.value("end_y", 1.0).toDouble();
    background_.centerX = settings.value("center_x", 0.5).toDouble();
    background_.centerY = settings.value("center_y", 0.5).toDouble();
    background_.radius = settings.value("radius", 1.0).toDouble();
    background_.imagePath = settings.value("image_path", "").toString();
    settings.endGroup();

    // 读取毛玻璃配置
    settings.beginGroup("frosted");
    frosted_.enabled = settings.value("enabled", true).toBool();
    frosted_.blurRadius = settings.value("blur_radius", 25).toInt();
    frosted_.opacity = settings.value("opacity", 200).toInt();
    frosted_.tintColor = QColor(settings.value("tint_color", "#FFFFFF").toString());
    settings.endGroup();

    // 读取主题配置
    settings.beginGroup("theme");
    theme_.primaryColor = QColor(settings.value("primary_color", "#4CAF50").toString());
    theme_.accentColor = QColor(settings.value("accent_color", "#45a049").toString());
    theme_.textColor = QColor(settings.value("text_color", "#FFFFFF").toString());

    QString borderColorStr = settings.value("border_color", "rgba(255, 255, 255, 100)").toString();
    theme_.borderColor = parseColorString(borderColorStr);

    QString panelBgStr = settings.value("panel_background", "rgba(255, 255, 255, 150)").toString();
    theme_.panelBackground = parseColorString(panelBgStr);
    settings.endGroup();

    return true;
}

bool StyleConfig::saveToFile(const QString& filePath) {
    QString fullPath = filePath;
    if (QFileInfo(filePath).isRelative()) {
        fullPath = QApplication::applicationDirPath() + "/" + filePath;
    }

    // 确保目录存在
    QFileInfo fileInfo(fullPath);
    QDir dir = fileInfo.absoluteDir();
    if (!dir.exists()) {
        dir.mkpath(".");
    }

    QSettings settings(fullPath, QSettings::IniFormat);
    #if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    settings.setIniCodec("UTF-8");
    #endif

    // 写入通用设置
    settings.beginGroup("general");
    settings.setValue("enable_frosted", enableFrosted_);
    settings.setValue("enable_custom_background", enableCustomBackground_);
    settings.endGroup();

    // 写入背景配置
    settings.beginGroup("background");
    QString typeStr;
    switch (background_.type) {
        case BackgroundType::Solid: typeStr = "solid"; break;
        case BackgroundType::Gradient: typeStr = "gradient"; break;
        case BackgroundType::Image: typeStr = "image"; break;
    }
    settings.setValue("type", typeStr);

    settings.setValue("gradient_type",
        background_.gradientType == GradientType::Linear ? "linear" : "radial");
    settings.setValue("solid_color", background_.solidColor.name());
    settings.setValue("start_color", background_.startColor.name());
    settings.setValue("end_color", background_.endColor.name());
    settings.setValue("start_x", background_.startX);
    settings.setValue("start_y", background_.startY);
    settings.setValue("end_x", background_.endX);
    settings.setValue("end_y", background_.endY);
    settings.setValue("center_x", background_.centerX);
    settings.setValue("center_y", background_.centerY);
    settings.setValue("radius", background_.radius);
    settings.setValue("image_path", background_.imagePath);
    settings.endGroup();

    // 写入毛玻璃配置
    settings.beginGroup("frosted");
    settings.setValue("enabled", frosted_.enabled);
    settings.setValue("blur_radius", frosted_.blurRadius);
    settings.setValue("opacity", frosted_.opacity);
    settings.setValue("tint_color", frosted_.tintColor.name());
    settings.endGroup();

    // 写入主题配置
    settings.beginGroup("theme");
    settings.setValue("primary_color", theme_.primaryColor.name());
    settings.setValue("accent_color", theme_.accentColor.name());
    settings.setValue("text_color", theme_.textColor.name());
    settings.setValue("border_color",
        QString("rgba(%1, %2, %3, %4)")
            .arg(theme_.borderColor.red())
            .arg(theme_.borderColor.green())
            .arg(theme_.borderColor.blue())
            .arg(theme_.borderColor.alpha()));
    settings.setValue("panel_background",
        QString("rgba(%1, %2, %3, %4)")
            .arg(theme_.panelBackground.red())
            .arg(theme_.panelBackground.green())
            .arg(theme_.panelBackground.blue())
            .arg(theme_.panelBackground.alpha()));
    settings.endGroup();

    settings.sync();
    return settings.status() == QSettings::NoError;
}

// 辅助函数：解析颜色字符串（支持 rgba 格式）
QColor StyleConfig::parseColorString(const QString& str) {
    QString trimmed = str.trimmed();

    // 尝试解析 rgba(r, g, b, a) 格式
    if (trimmed.startsWith("rgba(", Qt::CaseInsensitive) && trimmed.endsWith(")")) {
        QString values = trimmed.mid(5, trimmed.length() - 6);
        #if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
        QStringList parts = values.split(",", Qt::SkipEmptyParts);
        #else
        QStringList parts = values.split(",", QString::SkipEmptyParts);
        #endif

        if (parts.size() == 4) {
            int r = parts[0].trimmed().toInt();
            int g = parts[1].trimmed().toInt();
            int b = parts[2].trimmed().toInt();
            int a = parts[3].trimmed().toInt();
            return QColor(r, g, b, a);
        }
    }

    // 回退到标准颜色解析（#RGB, #RRGGBB）
    return QColor(trimmed);
}

}  // namespace gui
}  // namespace facesr
