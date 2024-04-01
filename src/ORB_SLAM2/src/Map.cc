#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"

namespace ORB_SLAM2_ROS2 {
/**
 * @brief 插入关键帧
 *
 * @param pKf   输入的要插入地图的关键帧
 * @param pMap  包装的this的共享指针
 */
void Map::insertKeyFrame(KeyFramePtr pKf, Map::SharedPtr pMap) {
    pKf->updateConnections();
    mvpKeyFrames.push_back(pKf);
    pKf->setMap(pMap);
}

/**
 * @brief 插入地图点
 *
 * @param pMp   输入的要插入地图的地图点
 * @param pMap  包装的this的共享指针
 */
void Map::insertMapPoint(MapPointPtr pMp, Map::SharedPtr pMap) {
    if (pMp->isInMap())
        return;
    mvpMapPoints.push_back(pMp);
    pMp->setMap(pMap);
}

} // namespace ORB_SLAM2_ROS2