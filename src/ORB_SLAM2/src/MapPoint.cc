#include <cmath>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 添加地图向关键帧的观测信息
 *
 * @param pKf       观测的关键帧
 * @param featId    关键帧对应特征点的id
 */
void MapPoint::addObservation(KeyFrameWeakPtr pKf, std::size_t featId) {
    assert(pKf.lock() && "关键帧为空");
    mObs.push_back(std::make_pair(pKf, featId));
}

/**
 * @brief 跟踪线程初始化地图专用函数，用于添加地图点的属性信息
 *
 * @param pRefKf    参考关键帧
 * @param nFeatId   特征点id（二维关键帧）
 */
void MapPoint::addAttriInit(KeyFramePtr pRefKf, std::size_t nFeatId) {
    mpRefKf = pRefKf;
    if (!pRefKf || pRefKf->isBad()) {
        mbIsBad = true;
        return;
    }
    auto kp = pRefKf->getLeftKeyPoints()[nFeatId];
    mObs.push_back(std::make_pair(pRefKf, nFeatId));
    pRefKf->getDescriptor()[nFeatId].copyTo(mDescriptor);
    cv::Mat center2Mp = mPoint3d - pRefKf->getFrameCenter();
    center2Mp.copyTo(mViewDirection);
    float x = center2Mp.at<float>(0, 0);
    float y = center2Mp.at<float>(1, 0);
    float z = center2Mp.at<float>(2, 0);
    float dist = std::sqrt(x * x + y * y + z * z);
    mnMaxDistance = dist * std::pow(ORBExtractor::mfScaledFactor, kp.octave - 0);
    mnMinDistance = dist * std::pow(ORBExtractor::mfScaledFactor, kp.octave - ORBExtractor::mnLevels + 1);
}

MapPoint::MapPoint(cv::Mat mP3d, KeyFrame::SharedPtr pRefKf, std::size_t nObs)
    : mPoint3d(std::move(mP3d)) {
    mId = mnNextId;
    addAttriInit(pRefKf, nObs);
}

MapPoint::MapPoint(cv::Mat mp3d)
    : mPoint3d(mp3d) {
    mId = mnNextId;
}

/**
 * @brief 判断某一位姿是否能观测到该地图点
 * @details
 *      1. 转换到相机坐标系下，z值大于0
 *      2. 投影到像素坐标系下，x和y值在图像上
 *      3. 长度在最大值和最小值之间
 *      4. 地图点平均观测角度和当前观测角度要小于60°
 *      5. 考虑到关键帧有可能失效，这里不做考虑，传入指针前应保证关键帧是有效的
 * @param pFrame        输入的帧
 * @param vecDistance   输出的计算pFrame光心到地图点的距离
 * @param uv            输出的投影产生的像素坐标系下的地图点
 * @return true     在可视范围内
 * @return false    不在可视范围内
 */
bool MapPoint::isInVision(VirtualFramePtr pFrame, float &vecDistance, cv::Point2f &uv, float &cosTheta) {
    cv::Mat Tcw;
    pFrame->getPose().copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    cv::Mat p3dC = Rcw * mPoint3d + tcw;
    if (p3dC.at<float>(2, 0) < 0)
        return false;
    float x = p3dC.at<float>(0, 0);
    float y = p3dC.at<float>(1, 0);
    float z = p3dC.at<float>(2, 0);
    float distance = std::sqrt(x * x + y * y + z * z);
    vecDistance = distance;
    if (distance > mnMaxDistance || distance < mnMinDistance)
        return false;
    float u = x / z * Camera::mfFx + Camera::mfCx;
    float v = y / z * Camera::mfFy + Camera::mfCy;
    uv.x = u;
    uv.y = v;
    if (u > pFrame->mnMaxU || u < 0 || v > pFrame->mnMaxV || v < 0)
        return false;
    cv::Mat viewDirection = Rcw * mViewDirection;
    float viewDirectionAbs = cv::norm(viewDirection, cv::NORM_L2);
    cosTheta = viewDirection.dot(p3dC) / (distance * viewDirectionAbs);
    if (cosTheta < 0.5)
        return false;
    return true;
}

/**
 * @brief 判断地图点是否是bad（即将要删除的点）
 *
 * @return true     判断外点，即将要删除
 * @return false    判断内点，不会删除
 */
bool MapPoint::isBad() const {
    std::unique_lock<std::mutex> lock(mBadMutex);
    return mbIsBad;
}

/**
 * @brief 金字塔层级预测
 * 在帧位姿相对稳定的时候使用，不稳定的时候，无意义，因为距离不准确
 * @param distance 输入的距离（待匹配光心到地图点的距离）
 * @return int 金字塔层级
 */
int MapPoint::predictLevel(float distance) const {
    int level = cvRound(std::log(mnMaxDistance / distance) / std::log(ORBExtractor::mfScaledFactor));
    if (level < 0)
        level = 0;
    else if (level > 7)
        level = 7;
    return level;
}

/**
 * @brief 由关键帧构造的地图点
 * @details
 *      1. 确定参考关键帧
 *      2. 最大和最小深度
 *      3. 被观测方向
 *      4. 地图点的代表描述子
 * @param mP3dW 世界坐标系下的地图点位置
 * @param pKf   参考关键帧（基于哪个关键帧构建的）
 * @param nObs  该地图点对应的2D关键点的索引
 * @return MapPoint::SharedPtr 返回的地图点
 */
MapPoint::SharedPtr MapPoint::create(cv::Mat mP3dW, KeyFrame::SharedPtr pKf, std::size_t nObs) {
    MapPoint::SharedPtr pMpoint(new MapPoint(mP3dW, pKf, nObs));
    ++mnNextId;
    return pMpoint;
}

/**
 * @brief 由普通帧构造的地图点
 * @details
 *      没有复杂的和关键帧之间的属性信息，只有位置信息
 * @param mP3dW 世界坐标系下的坐标
 * @return MapPoint::SharedPtr 返回构造成功的地图点共享指针
 */
MapPoint::SharedPtr MapPoint::create(cv::Mat mP3dW) {
    MapPoint::SharedPtr pMpoint(new MapPoint(mP3dW));
    ++mnNextId;
    return pMpoint;
}

unsigned int MapPoint::mnNextId = 0;

} // namespace ORB_SLAM2_ROS2
