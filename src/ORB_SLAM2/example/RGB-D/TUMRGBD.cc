#include <fstream>
#include <sstream>

#include <DBoW3/DBoW3.h>
#include <fmt/format.h>
#include <rclcpp/rclcpp.hpp>

#include "ORB_SLAM2/System.h"

using namespace std::chrono_literals;
using namespace ORB_SLAM2_ROS2;

/// 加载TUM数据集
void LoadImages(const std::string &Association, std::vector<std::string> &ColorImages,
                std::vector<std::string> &DepthImages, std::vector<double> &TimeStamps);

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: tum_example tum_data_seq tum_config_path association_path" << std::endl;
        return 1;
    }
    std::vector<double> TimeStamps;
    std::vector<std::string> ColorImages, DepthImages;
    LoadImages(argv[3], ColorImages, DepthImages, TimeStamps);
    rclcpp::init(argc, argv);
    auto system = std::make_shared<System>(argv[2]);
    for (std::size_t idx = 0, num = ColorImages.size(); idx < num; ++idx) {
        cv::Mat ColorImage = cv::imread(fmt::format("{}/{}", argv[1], ColorImages[idx]), cv::IMREAD_UNCHANGED);
        cv::Mat DepthImage = cv::imread(fmt::format("{}/{}", argv[1], DepthImages[idx]), cv::IMREAD_UNCHANGED);
        system->EstimatePose(ColorImage, DepthImage);
        std::this_thread::sleep_for(20ms);
    }
    rclcpp::shutdown();
    return 0;
}

/**
 * @brief 加载TUM数据集
 *
 * @param Association   输入的由associate.py 生成的匹配文件
 * @param ColorImages   输出的彩色图像路径集合
 * @param DepthImages   输出的深度图像路径集合
 * @param TimeStamps    输出的时间戳集合
 */
void LoadImages(const std::string &Association, std::vector<std::string> &ColorImages,
                std::vector<std::string> &DepthImages, std::vector<double> &TimeStamps) {
    std::ifstream ifs(Association);
    std::string lineStr;
    while (getline(ifs, lineStr)) {
        std::istringstream sstream(lineStr);
        double t;
        std::string cpath, dpath;
        sstream >> t >> cpath >> t >> dpath;
        ColorImages.push_back(cpath);
        DepthImages.push_back(dpath);
        TimeStamps.push_back(t);
    }
}