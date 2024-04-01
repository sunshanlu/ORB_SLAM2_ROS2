#include <string>

#include <fmt/format.h>
#include <opencv2/opencv.hpp>

#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/Tracking.h"

using namespace ORB_SLAM2_ROS2;

void readImage(cv::Mat &leftImg, cv::Mat &rightImg, int frameId) {
    static std::string fp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00";
    leftImg = cv::imread(fmt::format("{}/image_0/{:06d}.png", fp, frameId), cv::IMREAD_GRAYSCALE);
    rightImg = cv::imread(fmt::format("{}/image_1/{:06d}.png", fp, frameId), cv::IMREAD_GRAYSCALE);
    ++frameId;
}

int main() {
    cv::Mat leftImg, rightImg;
    auto pMap = std::make_shared<Map>();
    auto pKfDB = std::make_shared<KeyFrameDB>(Frame::mpVoc->size());
    auto tracking = std::make_shared<Tracking>(pMap, pKfDB);
    std::cout << std::endl;
    for (int idx = 0; idx < 20; ++idx) {
        readImage(leftImg, rightImg, idx);
        cv::Mat pose = tracking->grabFrame(leftImg, rightImg);
        if (idx % 5 == 0 && idx != 0) {
            auto kf = tracking->updateCurrFrame();
            for (std::size_t jdx = 0; jdx < kf->getMapPoints().size(); ++jdx) {
                MapPoint::SharedPtr pMp = kf->getMapPoints()[jdx];
                if (!pMp || pMp->isBad())
                    continue;
                pMp->addAttriInit(kf, jdx);
                pMap->insertMapPoint(pMp, pMap);
                pMap->insertKeyFrame(kf, pMap);
            }
            pKfDB->addKeyFrame(kf);
        }
        if (idx == 19) {
            idx = 9;
        }
        if (idx == 10) {
            std::cout << pose << std::endl;
        }
        std::cout << idx << std::endl;
    }

    return 0;
}
