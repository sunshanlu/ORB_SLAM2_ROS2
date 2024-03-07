#pragma once
#include <stdexcept>

#include <rclcpp/rclcpp.hpp>

namespace ORB_SLAM2_ROS2 {
/**
 * @brief 提取特征点数量较少错误（运行时的错误）
 *
 */
class FeatureLessError : public std::runtime_error {
public:
    explicit FeatureLessError(const std::string &what)
        : std::runtime_error(what) {
        RCLCPP_ERROR(rclcpp::get_logger("ORB_SLAM2"), what.c_str());
    }
};

/**
 * @brief 文件无法打开错误（运行时的错误）
 *
 */
class FileNotOpenError : public std::runtime_error {
public:
    explicit FileNotOpenError(const std::string &what)
        : std::runtime_error(what) {
        RCLCPP_ERROR(rclcpp::get_logger("ORB_SLAM2"), what.c_str());
    }
};

class ImageSizeError : public std::runtime_error {
public:
    explicit ImageSizeError(const std::string &what)
        : std::runtime_error(what) {
        RCLCPP_ERROR(rclcpp::get_logger("ORB_SLAM2"), what.c_str());
    }
};
} // namespace ORB_SLAM2_ROS2