#include <string>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Optimizer.h"
#include "ORB_SLAM2/PnPSolver.h"

void printMat(cv::Mat mat) { std::cout << mat << std::endl; }

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

    ORBMatcher matcher(0.7, true);
    int nMatches = matcher.searchByBow(pFrame, pKframe, matches);
    pFrame->setPose(pFrame2->getPose());
    int nInlier = Optimizer::OptimizePoseOnly(pFrame);
    std::cout << std::endl;
    std::cout << pFrame->getPose() << std::endl;

    // ORBMatcher::showMatches(pFrame->getLeftImage(), pFrame2->getLeftImage(), pFrame->getLeftKeyPoints(),
    // pFrame2->getLeftKeyPoints(), matches);

    // EPnP测试代码
    std::vector<cv::Mat> vMapPoints;
    std::vector<cv::KeyPoint> vORBPoints;
    std::vector<cv::Point2f> vpoints;
    std::vector<cv::Point3f> vMpoints;

    auto allMapPoints = pFrame->getMapPoints();
    auto allORBPoints = pFrame->getLeftKeyPoints();

    nMatches = 0;
    for (std::size_t idx = 0; idx < allORBPoints.size(); ++idx) {
        auto pMp = allMapPoints[idx];
        if (pMp && !pMp->isBad()) {
            cv::Mat pos = pMp->getPos();
            vMapPoints.push_back(pos);
            vORBPoints.push_back(allORBPoints[idx]);
            vpoints.push_back(allORBPoints[idx].pt);
            vMpoints.emplace_back(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
            ++nMatches;
        }
    }
    PnPRet modelRet;
    bool bNoMore;
    std::vector<std::size_t> inlierIndices;
    auto solver = PnPSolver::create(vMapPoints, vORBPoints);
    bool ret = solver->iterate(500, modelRet, bNoMore, inlierIndices);
    std::cout << "使用自定义的EPnP算法计算的相机位姿" << std::endl;
    std::cout << modelRet.mRcw << std::endl;
    std::cout << modelRet.mtcw << std::endl;

    cv::Mat R, t;
    cv::Mat cameraMatrix =
        (cv::Mat_<float>(3, 3) << Camera::mfFx, 0, Camera::mfCx, 0, Camera::mfFy, Camera::mfCy, 0, 0, 1);
    cv::solvePnP(vMpoints, vpoints, cameraMatrix, cv::noArray(), R, t, false, cv::SOLVEPNP_EPNP);
    std::cout << "使用OpenCV的EPnP算法计算的相机位姿" << std::endl;
    std::cout << R << std::endl;
    std::cout << t << std::endl;
    return 0;
}