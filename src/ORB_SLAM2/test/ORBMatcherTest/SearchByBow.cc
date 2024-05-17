#include <string>

#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/ORBMatcher.h"

using namespace ORB_SLAM2_ROS2;
int main() {
    std::string lStr0 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000000.png";
    std::string rStr0 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000000.png";
    std::string lStr1 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000001.png";
    std::string rStr1 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000001.png";
    std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    auto pFrame0 = Frame::create(cv::imread(lStr0, 0), cv::imread(rStr0, 0), 3000, briefFp, 20, 7);
    auto pFrame1 = Frame::create(cv::imread(lStr1, 0), cv::imread(rStr1, 0), 3000, briefFp, 20, 7);

    std::vector<MapPoint::SharedPtr> mapPoints;
    std::vector<cv::DMatch> matches;

    pFrame0->setPose(cv::Mat::eye(4, 4, CV_32F));
    pFrame0->unProject(mapPoints);
    auto pKframe0 = KeyFrame::create(*pFrame0);

    ORBMatcher matcher(0.7, true);
    int nMatches = matcher.searchByBow(pFrame1, pKframe0, matches, false, true);
    ORBMatcher::showMatches(pFrame1->getLeftImage(), pFrame0->getLeftImage(), pFrame1->getLeftKeyPoints(),
                            pFrame0->getLeftKeyPoints(), matches);

    return 0;
}