#pragma once
/**
 * @file config_parser.h
 * @brief 配置文件解析器
 *
 * 支持简单的INI风格配置文件解析
 *
 * 本解析器只做轻量级 key-value 读取：
 * [section] 下的 key 会保存成 "section.key"。
 * TrainConfig 依赖这个约定把 config/train_config.ini 映射到结构体字段。
 */

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include "logger.h"

namespace facesr {

/**
 * @brief 简单配置文件解析器
 *
 * 支持INI风格的配置文件格式:
 * [section]
 * key = value
 */
class ConfigParser {
public:
    /**
     * @brief 从文件加载配置
     */
    bool loadFromFile(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open config file: ", filepath);
            return false;
        }

        std::string line;
        std::string current_section;
        int line_num = 0;

        while (std::getline(file, line)) {
            line_num++;
            line = trim(line);

            // 跳过空行和注释
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }

            // 解析节
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.size() - 2);
                current_section = trim(current_section);
                continue;
            }

            // 解析键值对
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) {
                LOG_WARN("Invalid config line ", line_num, ": ", line);
                continue;
            }

            std::string key = trim(line.substr(0, eq_pos));
            std::string value = trim(line.substr(eq_pos + 1));

            // 移除引号
            if (value.size() >= 2) {
                if ((value.front() == '"' && value.back() == '"') ||
                    (value.front() == '\'' && value.back() == '\'')) {
                    value = value.substr(1, value.size() - 2);
                }
            }

            std::string full_key = current_section.empty() ? key : current_section + "." + key;
            data_[full_key] = value;
        }

        filepath_ = filepath;
        LOG_INFO("Loaded config from: ", filepath);
        return true;
    }

    /**
     * @brief 保存配置到文件
     */
    bool saveToFile(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            LOG_ERROR("Failed to create config file: ", filepath);
            return false;
        }

        std::map<std::string, std::map<std::string, std::string>> sections;

        for (const auto& [key, value] : data_) {
            size_t dot_pos = key.find('.');
            if (dot_pos != std::string::npos) {
                std::string section = key.substr(0, dot_pos);
                std::string subkey = key.substr(dot_pos + 1);
                sections[section][subkey] = value;
            } else {
                sections[""][key] = value;
            }
        }

        // 先写入无节的键值对
        if (sections.count("")) {
            for (const auto& [key, value] : sections[""]) {
                file << key << " = " << value << "\n";
            }
            file << "\n";
        }

        // 写入各节
        for (const auto& [section, values] : sections) {
            if (section.empty()) continue;
            file << "[" << section << "]\n";
            for (const auto& [key, value] : values) {
                file << key << " = " << value << "\n";
            }
            file << "\n";
        }

        LOG_INFO("Saved config to: ", filepath);
        return true;
    }

    // 获取字符串值
    std::string getString(const std::string& key, const std::string& default_val = "") const {
        auto it = data_.find(key);
        return (it != data_.end()) ? it->second : default_val;
    }

    // 获取整数值
    int getInt(const std::string& key, int default_val = 0) const {
        auto it = data_.find(key);
        if (it == data_.end()) return default_val;
        try {
            return std::stoi(it->second);
        } catch (...) {
            return default_val;
        }
    }

    // 获取浮点值
    double getDouble(const std::string& key, double default_val = 0.0) const {
        auto it = data_.find(key);
        if (it == data_.end()) return default_val;
        try {
            return std::stod(it->second);
        } catch (...) {
            return default_val;
        }
    }

    // 获取布尔值
    bool getBool(const std::string& key, bool default_val = false) const {
        auto it = data_.find(key);
        if (it == data_.end()) return default_val;
        std::string val = it->second;
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        return (val == "true" || val == "1" || val == "yes" || val == "on");
    }

    // 设置值
    void set(const std::string& key, const std::string& value) {
        data_[key] = value;
    }

    void set(const std::string& key, int value) {
        data_[key] = std::to_string(value);
    }

    void set(const std::string& key, double value) {
        data_[key] = std::to_string(value);
    }

    void set(const std::string& key, bool value) {
        data_[key] = value ? "true" : "false";
    }

    // 检查键是否存在
    bool hasKey(const std::string& key) const {
        return data_.find(key) != data_.end();
    }

    // 清空配置
    void clear() {
        data_.clear();
    }

    // 获取所有键
    std::vector<std::string> getKeys() const {
        std::vector<std::string> keys;
        keys.reserve(data_.size());
        for (const auto& [key, _] : data_) {
            keys.push_back(key);
        }
        return keys;
    }

private:
    static std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }

    std::map<std::string, std::string> data_;
    std::string filepath_;
};

}  // namespace facesr
