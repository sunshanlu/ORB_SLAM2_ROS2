#include "ORB_SLAM2/Tracking.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 处理普通帧
 *
 * @param leftImg   左图像
 * @param rightImg  右图像
 */
void Tracking::grabFrame(cv::Mat leftImg, cv::Mat rightImg) {
    mpCurrFrame = Frame::create(leftImg, rightImg, mnInitFeatures, msBriefTemFp, mnMaxThresh, mnMinThresh);
    if (mStatus == TrackingState::NOT_IMAGE_YET) {
        mStatus = TrackingState::NOT_INITING;
        auto pose = cv::Mat::eye(4, 4, CV_32F);
        mpCurrFrame->setPose(pose);
    }

    if (mStatus == TrackingState::NOT_INITING && mpCurrFrame->getN() >= 500)
        initForStereo();
    else
        return;
}

/**
 * @brief 双目相机的初始化
 *
 */
void Tracking::initForStereo() {
    std::vector<MapPoint::SharedPtr> mapPoints;
    mpCurrFrame->unProject(mapPoints);
    auto kfInit = KeyFrame::create(*mpCurrFrame);

    mpMap->insertKeyFrame(kfInit, mpMap);
    for (auto &pMp : mapPoints) {
        mpMap->insertMapPoint(pMp, mpMap);
    }
    mStatus = TrackingState::OK;
}

} // namespace ORB_SLAM2_ROS2