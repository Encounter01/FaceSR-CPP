#pragma once
/**
 * @file random.h
 * @brief 线程安全的随机数生成器
 *
 * 提供可复现的、线程安全的随机数生成功能
 */

#include <random>
#include <mutex>
#include <thread>

namespace facesr {

/**
 * @brief 线程安全的随机数生成器
 *
 * 使用thread_local确保每个线程有独立的随机数生成器
 */
class RandomGenerator {
public:
    /**
     * @brief 获取当前线程的随机数生成器
     */
    static RandomGenerator& getInstance() {
        thread_local RandomGenerator instance;
        return instance;
    }

    // 禁用拷贝和移动
    RandomGenerator(const RandomGenerator&) = delete;
    RandomGenerator& operator=(const RandomGenerator&) = delete;
    RandomGenerator(RandomGenerator&&) = delete;
    RandomGenerator& operator=(RandomGenerator&&) = delete;

    /**
     * @brief 设置随机种子
     * @param seed 种子值，如果为0则使用随机设备生成
     */
    void setSeed(unsigned int seed = 0) {
        if (seed == 0) {
            std::random_device rd;
            seed = rd();
        }
        generator_.seed(seed);
        seed_ = seed;
    }

    /**
     * @brief 获取当前种子
     */
    unsigned int getSeed() const {
        return seed_;
    }

    /**
     * @brief 生成[0, 1)范围内的均匀分布随机数
     */
    double uniform() {
        return uniform_dist_(generator_);
    }

    /**
     * @brief 生成[min, max)范围内的均匀分布随机数
     */
    double uniform(double min, double max) {
        return min + uniform() * (max - min);
    }

    /**
     * @brief 生成[min, max]范围内的均匀分布整数
     */
    int uniformInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator_);
    }

    /**
     * @brief 生成正态分布随机数
     */
    double normal(double mean = 0.0, double stddev = 1.0) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(generator_);
    }

    /**
     * @brief 以给定概率返回true
     */
    bool bernoulli(double probability = 0.5) {
        return uniform() < probability;
    }

    /**
     * @brief 获取底层的随机数生成器
     */
    std::mt19937& getGenerator() {
        return generator_;
    }

private:
    RandomGenerator() {
        std::random_device rd;
        seed_ = rd();
        generator_.seed(seed_);
    }

    std::mt19937 generator_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
    unsigned int seed_;
};

/**
 * @brief 全局随机数生成便捷函数
 */
namespace random {

/**
 * @brief 设置全局随机种子
 */
inline void seed(unsigned int seed = 0) {
    RandomGenerator::getInstance().setSeed(seed);
}

/**
 * @brief 生成[0, 1)范围内的均匀分布随机数
 */
inline double uniform() {
    return RandomGenerator::getInstance().uniform();
}

/**
 * @brief 生成[min, max)范围内的均匀分布随机数
 */
inline double uniform(double min, double max) {
    return RandomGenerator::getInstance().uniform(min, max);
}

/**
 * @brief 生成[min, max]范围内的均匀分布整数
 */
inline int uniformInt(int min, int max) {
    return RandomGenerator::getInstance().uniformInt(min, max);
}

/**
 * @brief 生成正态分布随机数
 */
inline double normal(double mean = 0.0, double stddev = 1.0) {
    return RandomGenerator::getInstance().normal(mean, stddev);
}

/**
 * @brief 以给定概率返回true
 */
inline bool bernoulli(double probability = 0.5) {
    return RandomGenerator::getInstance().bernoulli(probability);
}

}  // namespace random

}  // namespace facesr
