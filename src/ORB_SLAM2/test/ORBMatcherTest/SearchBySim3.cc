#include <string>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Optimizer.h"
#include "ORB_SLAM2/Sim3Solver.h"

using namespace ORB_SLAM2_ROS2;
int main() {
    std::string lStr0 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000000.png";
    std::string rStr0 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000000.png";
    std::string lStr1 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000001.png";
    std::string rStr1 = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_1/000001.png";
    std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";

    auto map = std::make_shared<Map>();

    auto pFrame0 = Frame::create(cv::imread(lStr0, 0), cv::imread(rStr0, 0), 1000, briefFp, 20, 7);
    auto pFrame1 = Frame::create(cv::imread(lStr1, 0), cv::imread(rStr1, 0), 1000, briefFp, 20, 7);

    std::vector<MapPoint::SharedPtr> vMapPs0, vMapPs1;
    cv::Mat identity = cv::Mat::eye(4, 4, CV_32F);
    pFrame0->setPose(identity.clone());
    identity.at<float>(2, 3) = -0.6;
    pFrame1->setPose(identity);
    pFrame0->unProject(vMapPs0);

    std::vector<cv::DMatch> matches;
    ORBMatcher matcher(0.7, true);
    int nMatches = matcher.searchByProjection(pFrame1, pFrame0, matches, 30);
    pFrame1->unProject(vMapPs1);
    auto pkFrame0 = KeyFrame::create(*pFrame0);
    auto pkFrame1 = KeyFrame::create(*pFrame1);
    for (std::size_t idx = 0; idx < vMapPs0.size(); ++idx) {
        auto &pMp = vMapPs0[idx];
        if (pMp) {
            pMp->addAttriInit(pkFrame0, idx);
            map->insertMapPoint(pMp, map);
        }
    }
    for (std::size_t idx = 0; idx < vMapPs1.size(); ++idx) {
        auto &pMp = vMapPs1[idx];
        if (pMp) {
            pMp->addAttriInit(pkFrame1, idx);
            map->insertMapPoint(pMp, map);
        }
    }

    Sim3Ret modelRet;
    bool bNoMore = false;
    std::vector<bool> vbChoose(matches.size(), true);
    std::vector<std::size_t> vInlierIndices;
    auto solver = Sim3Solver::create(pkFrame0, pkFrame1, matches, vbChoose, true);
    bool ret = solver->iterate(100, modelRet, bNoMore, vInlierIndices);
    std::cout << std::endl << "使用自定义的Sim3算法计算的相机位姿" << std::endl;
    std::cout << modelRet.mRqp << std::endl;
    std::cout << modelRet.mtqp << std::endl;

    std::vector<cv::DMatch> newMatches, newnewMatches;
    for (std::size_t idx = 0; idx < vbChoose.size(); ++idx) {
        if (vbChoose[idx])
            newMatches.push_back(matches[idx]);
    }

    for (const auto &idx : vInlierIndices)
        newnewMatches.push_back(newMatches[idx]);

    int nM = matcher.searchBySim3(pkFrame1, pkFrame0, newnewMatches, modelRet, 5);

    std::cout << nM - newMatches.size() << std::endl;

    ORBMatcher::showMatches(pFrame1->getLeftImage(), pFrame0->getLeftImage(), pFrame1->getLeftKeyPoints(),
                            pFrame0->getLeftKeyPoints(), newnewMatches);

    return 0;
}