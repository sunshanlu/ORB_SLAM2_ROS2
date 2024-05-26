#include <rclcpp/rclcpp.hpp>

#include "ORB_SLAM2/System.h"

using namespace ORB_SLAM2_ROS2;

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<System>("");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}