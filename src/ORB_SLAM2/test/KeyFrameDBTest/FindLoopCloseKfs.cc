#include <memory>

#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/MapPoint.h"

using namespace ORB_SLAM2_ROS2;

int main() {
    std::string lstr0 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000000.png";
    std::string lstr1 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000001.png";
    std::string lstr2 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000002.png";
    std::string rstr0 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000000.png";
    std::string rstr1 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000001.png";
    std::string rstr2 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000002.png";
    std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    auto pFrame0 = Frame::create(cv::imread(lstr0, 0), cv::imread(rstr0, 0), 2000, briefFp, 20, 7);
    auto pFrame1 = Frame::create(cv::imread(lstr1, 0), cv::imread(rstr1, 0), 2000, briefFp, 20, 7);
    auto pFrame2 = Frame::create(cv::imread(lstr2, 0), cv::imread(rstr2, 0), 2000, briefFp, 20, 7);

    std::vector<cv::DMatch> matches;
    std::vector<MapPoint::SharedPtr> vMapPoints;
    pFrame0->setPose(cv::Mat::eye(4, 4, CV_32F));
    pFrame0->unProject(vMapPoints);
    auto pKframe0 = KeyFrame::create(*pFrame0);
    ORBMatcher matcher(0.8, true);
    matcher.searchByBow(pFrame1, pKframe0, matches);
    auto pKframe1 = KeyFrame::create(*pFrame1);
    auto pKframe2 = KeyFrame::create(*pFrame2);
    /// 构建关键帧0和1相连
    for (auto &m : matches) {
        auto pMp = pKframe0->getMapPoint(m.trainIdx);
        pMp->addObservation(pKframe0, m.trainIdx);
        pMp->addObservation(pKframe1, m.queryIdx);
    }
    KeyFrame::updateConnections(pKframe1);

    std::vector<KeyFrame::SharedPtr> vpCandidatesLoop;
    std::vector<KeyFrame::SharedPtr> vpCandidatesReloc;
    auto pKframeDB = std::make_shared<KeyFrameDB>(Frame::mpVoc->size());
    pKframeDB->addKeyFrame(pKframe0);
    pKframeDB->addKeyFrame(pKframe2);
    pKframeDB->findRelocKfs(pFrame1, vpCandidatesReloc);
    pKframeDB->findLoopCloseKfs(pKframe1, vpCandidatesLoop);
    std::cout << std::endl;
    return 0;
}
