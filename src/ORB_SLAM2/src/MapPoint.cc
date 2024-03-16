#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/KeyFrame.h"

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
 * @brief 判断地图点是否是bad（即将要删除的点）
 * 
 * @return true     判断外点，即将要删除
 * @return false    判断内点，不会删除
 */
bool MapPoint::isBad() const {
    std::unique_lock<std::mutex> lock(mBadMutex);
    return mbIsBad;
}

unsigned int MapPoint::mnNextId = 0;

} // namespace ORB_SLAM2_ROS2
