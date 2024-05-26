#include <fstream>

#include <DBoW3/DBoW3.h>
#include <fmt/format.h>
#include <rclcpp/rclcpp.hpp>

#include "ORB_SLAM2/System.h"

using namespace std::chrono_literals;
using namespace ORB_SLAM2_ROS2;

/// 加载KITTI数据集
void LoadImages(const std::string &SequencePath, std::vector<std::string> &Images, std::vector<double> &TimeStamps);

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: kitti_example kitti_data_seq kitti_config_path" << std::endl;
        return 1;
    }
    std::vector<std::string> Images;
    std::vector<double> TimeStamps;
    LoadImages(argv[1], Images, TimeStamps);
    rclcpp::init(argc, argv);
    auto system = std::make_shared<System>(argv[2]);

    for (int idx = 0, num = Images.size(); idx < num; ++idx) {
        cv::Mat lImage = cv::imread(fmt::format("{}/image_0/{}", argv[1], Images[idx]), cv::IMREAD_UNCHANGED);
        cv::Mat rImage = cv::imread(fmt::format("{}/image_1/{}", argv[1], Images[idx]), cv::IMREAD_UNCHANGED);
        system->EstimatePose(lImage, rImage);
        std::this_thread::sleep_for(5ms);
    }
    rclcpp::shutdown();
    return 0;
}

/**
 * @brief 加载KITTI数据集
 *
 * @param SequencePath  输入的数据集路径
 * @param Images        输出的图像文件名
 * @param TimeStamps    输出的时间戳集合
 */
void LoadImages(const std::string &SequencePath, std::vector<std::string> &Images, std::vector<double> &TimeStamps) {
    std::string lineStr;
    std::ifstream ifs(fmt::format("{}/times.txt", SequencePath));
    int ImageId = 0;
    while (std::getline(ifs, lineStr)) {
        double timeStamp;
        std::istringstream sstream(lineStr);
        sstream >> timeStamp;
        TimeStamps.push_back(timeStamp);
        Images.push_back(fmt::format("{:06d}.png", ImageId++));
    }
}