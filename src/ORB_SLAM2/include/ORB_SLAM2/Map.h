#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace ORB_SLAM2_ROS2 {
class MapPoint;
class KeyFrame;

class Map {
public:
    typedef std::shared_ptr<Map> SharedPtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<MapPoint> MapPointPtr;

    Map() = default;

    /// 向地图中插入关键帧
    void insertKeyFrame(KeyFramePtr pKf, SharedPtr map);

    /// 向地图中插入地图点
    void insertMapPoint(MapPointPtr pMp, SharedPtr map);

    /// 删除关键帧
    void eraseKeyFrame(KeyFramePtr pKf);

    /// 删除地图点
    void eraseMapPoint(MapPointPtr pMp);

    /// 获取地图中关键帧的数目
    std::size_t keyFramesInMap() const {
        std::unique_lock<std::mutex> lock(mKfMutex);
        return mspKeyFrames.size();
    }

    /// 获取地图中地图点的数目
    std::size_t mapPointsInMap() const {
        std::unique_lock<std::mutex> lock(mMpMutex);
        return mspMapPoints.size();
    }

    /// 获取地图中所有关键帧
    std::vector<KeyFramePtr> getAllKeyFrames() {
        std::vector<KeyFramePtr> mvpAllKfs;
        {
            std::unique_lock<std::mutex> lock(mKfMutex);
            std::copy(mspKeyFrames.begin(), mspKeyFrames.end(), std::back_inserter(mvpAllKfs));
        }
        return mvpAllKfs;
    }

    /// 获取所有地图点
    std::vector<MapPointPtr> getAllMapPoints() {
        std::vector<MapPointPtr> mvpAllMps;
        {
            std::unique_lock<std::mutex> lock(mMpMutex);
            std::copy(mspMapPoints.begin(), mspMapPoints.end(), std::back_inserter(mvpAllMps));
        }
        return mvpAllMps;
    }

    /// 设置更新标识
    void setUpdate(bool flag) {
        std::unique_lock<std::mutex> lock(mUpMutex);
        mbUpdate = flag;
    }

    /// 获取更新标识
    bool getUpdate() {
        std::unique_lock<std::mutex> lock(mUpMutex);
        return mbUpdate;
    }

private:
    std::set<KeyFramePtr> mspKeyFrames; ///< 地图中的所有关键帧
    std::set<MapPointPtr> mspMapPoints; ///< 地图中的所有地图点
    bool mbUpdate = false;              ///< 地图更新标志
    mutable std::mutex mMpMutex;        ///< 地图点互斥锁
    mutable std::mutex mKfMutex;        ///< 关键帧互斥锁
    mutable std::mutex mUpMutex;        ///< 维护地图更新表示的互斥锁
};
} // namespace ORB_SLAM2_ROS2
