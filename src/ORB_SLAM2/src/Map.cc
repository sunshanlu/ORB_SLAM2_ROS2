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
    {
        std::unique_lock<std::mutex> lock(mKfMutex);
        mspKeyFrames.insert(pKf);
        pKf->setMap(pMap);
    }
    setUpdate(true);
}

/**
 * @brief 插入地图点
 *
 * @param pMp   输入的要插入地图的地图点
 * @param pMap  包装的this的共享指针
 */
void Map::insertMapPoint(MapPointPtr pMp, Map::SharedPtr pMap) {
    {
        std::unique_lock<std::mutex> lock(mMpMutex);
        if (pMp->isInMap())
            return;
        mspMapPoints.insert(pMp);
        pMp->setMap(pMap);
    }
    setUpdate(true);
}

/**
 * @brief 删除某个地图点
 *
 * @param pMp 输入的待删除的地图点
 */
void Map::eraseMapPoint(MapPointPtr pMp) {
    {
        std::unique_lock<std::mutex> lock(mMpMutex);
        auto it = mspMapPoints.find(pMp);
        if (it == mspMapPoints.end())
            return;
        mspMapPoints.erase(pMp);
        pMp->setMapNull();
    }
    setUpdate(true);
}

/**
 * @brief 删除某个关键帧
 *
 * @param pKf 输入的待删除的关键帧
 */
void Map::eraseKeyFrame(KeyFramePtr pKf) {
    {
        std::unique_lock<std::mutex> lock(mKfMutex);
        mspKeyFrames.erase(pKf);
        pKf->setMapNull();
    }
    setUpdate(true);
}

} // namespace ORB_SLAM2_ROS2