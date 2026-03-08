#pragma once
/**
 * @file logger.h
 * @brief 日志系统
 *
 * 提供统一的日志记录接口，支持不同日志级别和输出方式
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>

namespace facesr {

/**
 * @brief 日志级别
 */
enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4
};

/**
 * @brief 将日志级别转换为字符串
 */
inline const char* logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:   return "DEBUG";
        case LogLevel::Info:    return "INFO";
        case LogLevel::Warning: return "WARN";
        case LogLevel::Error:   return "ERROR";
        case LogLevel::Fatal:   return "FATAL";
        default: return "UNKNOWN";
    }
}

/**
 * @brief 日志记录器类
 *
 * 线程安全的单例日志记录器
 */
class Logger {
public:
    /**
     * @brief 获取日志记录器单例
     */
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    // 禁用拷贝和移动
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    /**
     * @brief 设置最小日志级别
     */
    void setLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        min_level_ = level;
    }

    /**
     * @brief 获取当前日志级别
     */
    LogLevel getLevel() const {
        return min_level_;
    }

    /**
     * @brief 设置日志文件输出
     */
    bool setLogFile(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
        file_.open(path, std::ios::out | std::ios::app);
        return file_.is_open();
    }

    /**
     * @brief 关闭日志文件
     */
    void closeLogFile() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
    }

    /**
     * @brief 启用/禁用控制台输出
     */
    void setConsoleOutput(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        console_output_ = enabled;
    }

    /**
     * @brief 启用/禁用时间戳
     */
    void setTimestamp(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        show_timestamp_ = enabled;
    }

    /**
     * @brief 记录日志
     */
    template<typename... Args>
    void log(LogLevel level, const char* file, int line, Args&&... args) {
        if (level < min_level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        std::ostringstream oss;

        // 时间戳
        if (show_timestamp_) {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
            oss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
                << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
        }

        // 日志级别
        oss << "[" << logLevelToString(level) << "] ";

        // 源文件位置（仅Debug和Error级别）
        if (level == LogLevel::Debug || level >= LogLevel::Error) {
            std::string filename(file);
            size_t pos = filename.find_last_of("/\\");
            if (pos != std::string::npos) {
                filename = filename.substr(pos + 1);
            }
            oss << "[" << filename << ":" << line << "] ";
        }

        // 消息内容
        ((oss << std::forward<Args>(args)), ...);
        oss << "\n";

        std::string message = oss.str();

        // 输出到控制台
        if (console_output_) {
            if (level >= LogLevel::Error) {
                std::cerr << message;
            } else {
                std::cout << message;
            }
        }

        // 输出到文件
        if (file_.is_open()) {
            file_ << message;
            file_.flush();
        }
    }

    /**
     * @brief 进度日志（覆盖同一行）
     */
    template<typename... Args>
    void progress(Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream oss;
        ((oss << std::forward<Args>(args)), ...);
        std::cout << "\r" << oss.str() << std::flush;
    }

    /**
     * @brief 结束进度日志行
     */
    void endProgress() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << std::endl;
    }

private:
    Logger() = default;
    ~Logger() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    std::mutex mutex_;
    LogLevel min_level_ = LogLevel::Info;
    std::ofstream file_;
    bool console_output_ = true;
    bool show_timestamp_ = true;
};

// ==================== 日志宏定义 ====================

#define LOG_DEBUG(...) \
    facesr::Logger::getInstance().log(facesr::LogLevel::Debug, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_INFO(...) \
    facesr::Logger::getInstance().log(facesr::LogLevel::Info, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_WARN(...) \
    facesr::Logger::getInstance().log(facesr::LogLevel::Warning, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_ERROR(...) \
    facesr::Logger::getInstance().log(facesr::LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_FATAL(...) \
    facesr::Logger::getInstance().log(facesr::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_PROGRESS(...) \
    facesr::Logger::getInstance().progress(__VA_ARGS__)

#define LOG_END_PROGRESS() \
    facesr::Logger::getInstance().endProgress()

}  // namespace facesr
