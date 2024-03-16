#include "ORB_SLAM2/Tracking.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Optimizer.h"

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
    else if (mStatus == TrackingState::OK) {
        bool bOK = false;
        mpCurrFrame->setPose(mpLastFrame->getPose());
        if (mVelocity.empty()) {
            bOK = trackReference();
        }else {
            bOK = trackMotionModel();
            if (!bOK)
                bOK = trackReference();
        }
        //todo: bOK为true时，跟踪局部地图，否则进行重定位
        
    }

    mpLastFrame = mpCurrFrame;
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
    for (std::size_t idx = 0; idx < mapPoints.size(); ++idx) {
        auto &pMp = mapPoints[idx];
        if (!pMp)
            continue;
        mpMap->insertMapPoint(pMp, mpMap);
        pMp->addObservation(kfInit, idx);
    }
    mStatus = TrackingState::OK;
    mpRefKf = kfInit;
}

/**
 * @brief 跟踪参考关键帧
 * @details
 *      1. 基于词袋的匹配要求成功匹配大于等于15
 *      2. 基于基于OptimizePoseOnly的优化，要求内点数目大于等于10
 * @return true     跟踪参考关键帧失败
 * @return false    跟踪参考关键帧成功
 */
bool Tracking::trackReference() {
    ORBMatcher matcher(0.7, true);
    std::vector<cv::DMatch> matches;
    int nMatches = matcher.searchByBow(mpCurrFrame, mpRefKf, matches);
    if (nMatches < 15) {
        return false;
    }
    int nInliers = Optimizer::OptimizePoseOnly(mpCurrFrame);

    return nInliers >= 10;
}

/**
 * @brief 基于恒速模型的跟踪
 * @details
 *      1. 基于重投影匹配，初始的半径为15，如果没有合适的匹配，寻找半径变为30
 *      2. OptimizePoseOnly做了外点的剔除（误差较大的边和投影超出图像边界的地图点）
 * @return true     跟踪成功
 * @return false    跟踪失败
 */
bool Tracking::trackMotionModel() {
    std::vector<cv::DMatch> matches;
    processLastFrame();
    mpCurrFrame->setPose(mVelocity * mpLastFrame->getPose());
    ORBMatcher matcher(0.7, true);
    int nMatches = matcher.searchByProjection(mpCurrFrame, mpLastFrame, matches, 15);
    if (nMatches < 20) {
        nMatches += matcher.searchByProjection(mpCurrFrame, mpLastFrame, matches, 30);
    }
    if (nMatches < 20) {
        return false;
    }
    int inLiers = Optimizer::OptimizePoseOnly(mpCurrFrame);
    return inLiers >= 10;
}

/**
 * @brief 处理上一帧
 * @details
 *      1. 利用参考关键帧进行上一帧的位姿纠正
 *      2. 对上一帧进行反投影，构造临时地图点
 *      3. 使用Tracking进行临时地图点的维护
 *      4. 值得注意的是，维护的临时地图点中，有nullptr
 */
void Tracking::processLastFrame() {
    auto refKf = mpLastFrame->getRefKF();
    assert(refKf && "上一帧的参考关键帧为空！");
    mpLastFrame->setPose(mTlr * refKf->getPose());

    std::vector<MapPoint::SharedPtr> mapPoints;
    mpLastFrame->unProject(mapPoints);
    std::swap(mapPoints, mvpTempMappoints);
}

} // namespace ORB_SLAM2_ROS2
