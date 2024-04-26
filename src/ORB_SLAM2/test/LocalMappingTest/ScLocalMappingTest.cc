/// 跟踪线程和局部建图线程串联测试
#include <string>

#include <fmt/format.h>
#include <opencv2/opencv.hpp>

#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/LocalMapping.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/Tracking.h"
#include "ORB_SLAM2/Viewer.h"

using namespace ORB_SLAM2_ROS2;

void readImage(cv::Mat &leftImg, cv::Mat &rightImg, const std::size_t &frameId) {
    static std::string fp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00";
    leftImg = cv::imread(fmt::format("{}/image_0/{:06d}.png", fp, frameId), cv::IMREAD_GRAYSCALE);
    rightImg = cv::imread(fmt::format("{}/image_1/{:06d}.png", fp, frameId), cv::IMREAD_GRAYSCALE);
}

int main() {
    cv::Mat leftImg, rightImg;
    auto pMap = std::make_shared<Map>();
    auto pKfDB = std::make_shared<KeyFrameDB>(Frame::mpVoc->size());
    auto tracker = std::make_shared<Tracking>(pMap, pKfDB);
    auto localMapper = std::make_shared<LocalMapping>(pMap);
    auto viewer = std::make_shared<Viewer>(pMap, tracker);
    tracker->setLocalMapper(localMapper);
    tracker->setViewer(viewer);
    std::thread vThread(&Viewer::run, viewer);

    std::cout << std::endl;
    for (int idx = 0; idx < 4541; ++idx) {
        readImage(leftImg, rightImg, idx);
        cv::Mat pose = tracker->grabFrame(leftImg, rightImg);
        localMapper->runOnce();
        localMapper->addKF2DB(pKfDB);
        std::cout << idx << std::endl;
        std::cout << pose << std::endl << std::endl;
    }
    vThread.join();
    return 0;
}