#include <string>

#include "ORB_SLAM2/Frame.h"

using namespace ORB_SLAM2_ROS2;

int main() {
    std::string leftImgFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000634.png";
    std::string rightImgFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000634.png";
    std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    cv::Mat leftImg = cv::imread(leftImgFp, cv::IMREAD_GRAYSCALE);
    cv::Mat rightImg = cv::imread(rightImgFp, cv::IMREAD_GRAYSCALE);
    Frame::SharedPtr pFrame = Frame::create(leftImg, rightImg, 2000, briefFp, 20, 7);
    pFrame->showStereoMatches();
    return 0;
}
