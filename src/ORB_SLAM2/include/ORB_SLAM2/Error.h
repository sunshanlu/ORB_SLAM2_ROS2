#pragma once
#include <stdexcept>

#include <rclcpp/rclcpp.hpp>

namespace ORB_SLAM2_ROS2 {

/**
 * @brief ORB_SLAM2系统运行时错误抽象类
 * 
 */
class ORBSlam2Error : public std::runtime_error {
public:
    ORBSlam2Error(const std::string &what)
        : std::runtime_error(what) {
        RCLCPP_ERROR(rclcpp::get_logger("ORB_SLAM2"), what.c_str());
    }

    /// 定义纯虚函数，避免被非继承构造
    virtual void pureVirtual() = 0;
};

/**
 * @brief 提取特征点数量较少错误（运行时的错误）
 *
 */
class FeatureLessError : public ORBSlam2Error {
public:
    explicit FeatureLessError(const std::string &what)
        : ORBSlam2Error(what) {}

    void pureVirtual() override {}
};

/**
 * @brief 文件无法打开错误（运行时的错误）
 *
 */
class FileNotOpenError : public ORBSlam2Error {
public:
    explicit FileNotOpenError(const std::string &what)
        : ORBSlam2Error(what) {}

    void pureVirtual() override {}
};

/**
 * @brief 图像尺寸错误
 * 
 */
class ImageSizeError : public ORBSlam2Error {
public:
    explicit ImageSizeError(const std::string &what)
        : ORBSlam2Error(what) {}

    void pureVirtual() override {}
};

/**
 * @brief 线程错误
 * 
 */
class ThreadError : public ORBSlam2Error {
public:
    explicit ThreadError(const std::string &what)
        : ORBSlam2Error(what) {}

    void pureVirtual() override {}
};

/**
 * @brief EPnP求解错误
 * 
 */
class EPnPError : public ORBSlam2Error {
public:
    explicit EPnPError(const std::string &what)
        : ORBSlam2Error(what) {}

    void pureVirtual() override {}
};

} // namespace ORB_SLAM2_ROS2