#include <string>

#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Optimizer.h"

using namespace ORB_SLAM2_ROS2;
int main() {
    std::string fLeftFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/001076.png";
    std::string fRightFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/001076.png";
    std::string kfLeftFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/001075.png";
    std::string kfRightFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/001075.png";
    std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    auto pFrame = Frame::create(cv::imread(fLeftFp, 0), cv::imread(fRightFp, 0), 2000, briefFp, 20, 7);
    auto pFrame2 = Frame::create(cv::imread(kfLeftFp, 0), cv::imread(kfRightFp, 0), 2000, briefFp, 20, 7);

    std::vector<MapPoint::SharedPtr> mapPoints;
    std::vector<cv::DMatch> matches;

    pFrame2->setPose(cv::Mat::eye(4, 4, CV_32F));
    pFrame2->unProject(mapPoints);
    cv::Mat pose =
        (cv::Mat_<float>(4, 4) << 0.99999118, 0.0036771602, -0.0019949719, -0.002166748, -0.0036713216, 0.99998897,
         0.0029149437, 0.010447502, 0.0020056842, -0.0029075935, 0.99999386, -1.0079346, 0, 0, 0, 1);
    pFrame->setPose(pose);

    ORBMatcher matcher(0.7, true);
    int nMatches = matcher.searchByProjection(pFrame, pFrame2, matches, 30);
    int nInliers = Optimizer::OptimizePoseOnly(pFrame);
    std::cout << std::endl << pFrame->getPose() << std::endl;
    std::cout << "nInliers: " << nInliers << std::endl;

    ORBMatcher::showMatches(pFrame->getLeftImage(), pFrame2->getLeftImage(), pFrame->getLeftKeyPoints(),
                            pFrame2->getLeftKeyPoints(), matches);

    return 0;
}