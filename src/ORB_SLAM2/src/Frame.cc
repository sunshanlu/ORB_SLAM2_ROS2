#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Error.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 展示双目图像的匹配结果
 *
 */
void Frame::showStereoMatches() const {
    cv::Mat showImage;
    std::vector<cv::Mat> imgs{mLeftIm, mRightIm};
    cv::hconcat(imgs, showImage);
    cv::cvtColor(showImage, showImage, cv::COLOR_GRAY2BGR);
    std::vector<cv::KeyPoint> rightKps, leftKps;
    for (std::size_t i = 0; i < mvFeatsLeft.size(); ++i) {
        const auto &rightU = mvFeatsRightU[i];
        if (rightU == -1) {
            continue;
        }
        const auto &lkp = mvFeatsLeft[i];
        cv::KeyPoint rkp;
        rkp.pt.x = rightU + mLeftIm.cols;
        rkp.pt.y = lkp.pt.y;
        cv::line(showImage, lkp.pt, rkp.pt, cv::Scalar(255, 0, 0));
        rightKps.push_back(rkp);
        leftKps.push_back(lkp);
    }
    cv::drawKeypoints(showImage, leftKps, showImage, cv::Scalar(0, 255, 0));
    cv::drawKeypoints(showImage, rightKps, showImage, cv::Scalar(0, 0, 255));
    cv::imshow("showImage", showImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

/**
 * @brief 普通帧的构造函数
 *
 * @param leftImg   左图图像
 * @param rightImg  右图图像
 * @param nFeatures 需要的特征点数
 * @param briefFp   BRIEF模板文件路径
 * @param maxThresh FAST检测的最大阈值
 * @param minThresh FAST检测的最小阈值
 */
Frame::Frame(cv::Mat leftImg, cv::Mat rightImg, int nFeatures, const std::string &briefFp, int maxThresh, int minThresh)
    : mLeftIm(leftImg)
    , mRightIm(rightImg) {
    mpExtractorLeft = std::make_shared<ORBExtractor>(mLeftIm, nFeatures, 8, 1.2, briefFp, maxThresh, minThresh);
    mpExtractorRight = std::make_shared<ORBExtractor>(mRightIm, nFeatures, 8, 1.2, briefFp, maxThresh, minThresh);

    std::thread leftThread(
        std::bind(&ORBExtractor::extract, mpExtractorLeft.get(), std::ref(mvFeatsLeft), std::ref(mvLeftDescriptor)));
    std::thread rightThread(
        std::bind(&ORBExtractor::extract, mpExtractorRight.get(), std::ref(mvFeatsRight), std::ref(mRightDescriptor)));
    if (!leftThread.joinable() || !rightThread.joinable())
        throw ThreadError("普通帧提取特征点时线程不可joinable");
    leftThread.join();
    rightThread.join();
    mvpMapPoints.resize(mvFeatsLeft.size(), nullptr);
    mnID = mnNextID;
}

/**
 * @brief 初始化地图点
 * @details
 *      1. 普通帧没有创建地图点的权限，只是利用普通帧的信息进行地图点计算
 * @param mapPoints 输出的地图点信息
 */
void Frame::unProject(std::vector<MapPoint::SharedPtr> &mapPoints) {
    mapPoints.clear();
    for (std::size_t idx = 0; idx < mvFeatsLeft.size(); ++idx) {
        assert(!mTcw.empty() && !mRwc.empty() && !mtwc.empty());
        cv::Mat p3dC = unProject(idx);
        if (!p3dC.empty()) {
            cv::Mat p3dW = mRwc * p3dC + mtwc;
            mapPoints.push_back(MapPoint::create(p3dW));
        } else {
            mapPoints.push_back(nullptr);
        }
    }
    mvpMapPoints = mapPoints;
}

/**
 * @brief 逆投影到相机坐标系中（根据特征点位置和深度）
 *
 * @param idx 左图特征点索引
 * @return cv::Mat 输出的相机坐标系下的3D点
 */
cv::Mat Frame::unProject(std::size_t idx) {
    const cv::KeyPoint &lKp = mvFeatsLeft[idx];
    const double &depth = mvDepths[idx];
    if (depth < 0)
        return cv::Mat();
    float x = (lKp.pt.x - Camera::mfCx) / Camera::mfFx;
    float y = (lKp.pt.y - Camera::mfCy) / Camera::mfFy;
    cv::Mat p3dC(3, 1, CV_32F);
    p3dC.at<float>(0) = depth * x;
    p3dC.at<float>(1) = depth * y;
    p3dC.at<float>(2) = depth;
    return p3dC;
}

std::string VirtualFrame::msVoc = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/Vocabulary/ORBvoc.txt";
VirtualFrame::VocabPtr VirtualFrame::mpVoc = std::make_shared<DBoW3::Vocabulary>(VirtualFrame::msVoc);
std::size_t Frame::mnNextID = 0;
} // namespace ORB_SLAM2_ROS2