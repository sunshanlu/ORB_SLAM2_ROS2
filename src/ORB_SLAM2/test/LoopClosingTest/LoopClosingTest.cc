/// 回环闭合线程测试
#include <string>

#include <fmt/format.h>
#include <opencv2/opencv.hpp>

#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/LocalMapping.h"
#include "ORB_SLAM2/LoopClosing.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/Tracking.h"
#include "ORB_SLAM2/Viewer.h"

using namespace ORB_SLAM2_ROS2;
using namespace std::chrono_literals;

void readImage(cv::Mat &leftImg, cv::Mat &rightImg, const std::size_t &frameId) {
    static std::string fp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00";
    leftImg = cv::imread(fmt::format("{}/image_0/{:06d}.png", fp, frameId), cv::IMREAD_GRAYSCALE);
    rightImg = cv::imread(fmt::format("{}/image_1/{:06d}.png", fp, frameId), cv::IMREAD_GRAYSCALE);
}

std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";

int main() {
    cv::Mat leftImg, rightImg;
    auto pMap = std::make_shared<Map>();
    auto pKfDB = std::make_shared<KeyFrameDB>(Frame::mpVoc->size());
    auto tracker0 = std::make_shared<Tracking>(pMap, pKfDB);
    auto localMapper = std::make_shared<LocalMapping>(pMap);
    auto viewer = std::make_shared<Viewer>(pMap, tracker0);
    auto loopClosing = std::make_shared<LoopClosing>(pKfDB, pMap, localMapper, tracker0);
    localMapper->setLoopClosing(loopClosing);

    tracker0->setLocalMapper(localMapper);
    tracker0->setViewer(viewer);

    std::thread lMapThread(std::bind(&LocalMapping::run, localMapper));
    std::thread viewerThread(std::bind(&Viewer::run, viewer));
    std::thread lCloseThread(std::bind(&LoopClosing::run, loopClosing));

    std::cout << std::endl;
    for (int idx = 0; idx < 4541; ++idx) {
        readImage(leftImg, rightImg, idx);
        cv::Mat pose = tracker0->grabFrame(leftImg, rightImg);
        // std::this_thread::sleep_for(30ms);
        // std::cout << idx << std::endl;
        // std::cout << pose << std::endl;
        cv::imshow("Tracking Window", leftImg);
        cv::waitKey(10);
    }
    cv::destroyAllWindows();

    lMapThread.join();
    viewerThread.join();
    lCloseThread.join();
    return 0;
}