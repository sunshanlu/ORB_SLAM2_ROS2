#include <memory>

#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/MapPoint.h"

using namespace ORB_SLAM2_ROS2;

int main() {
    std::string fLeftFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000001.png";
    std::string fRightFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000001.png";
    std::string kfLeftFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000000.png";
    std::string kfRightFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000000.png";
    std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    auto pFrame = Frame::create(cv::imread(fLeftFp, 0), cv::imread(fRightFp, 0), 2000, briefFp, 20, 7);
    auto pFrame2 = Frame::create(cv::imread(kfLeftFp, 0), cv::imread(kfRightFp, 0), 2000, briefFp, 20, 7);

    std::vector<MapPoint::SharedPtr> mapPoints;
    std::vector<cv::DMatch> matches;

    pFrame2->setPose(cv::Mat::eye(4, 4, CV_32F));
    pFrame2->unProject(mapPoints);
    auto pKframe = KeyFrame::create(*pFrame2);

    std::vector<KeyFrame::SharedPtr> vpCandidates;
    auto pKframeDB = std::make_shared<KeyFrameDB>(Frame::mpVoc->size());
    pKframeDB->addKeyFrame(pKframe);
    pKframeDB->findRelocKfs(pFrame, vpCandidates);
    std::cout << std::endl;
    std::cout << "Candidates: " << vpCandidates.size() << std::endl;
    return 0;
}
