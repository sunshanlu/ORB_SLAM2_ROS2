#pragma once

#include <memory>
#include <vector>

namespace ORB_SLAM2_ROS2 {
class MapPoint;
class KeyFrame;

class Map {
public:
    typedef std::shared_ptr<Map> SharedPtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<MapPoint> MapPointPtr;

    /// 向地图中插入关键帧
    void insertKeyFrame(KeyFramePtr pKf, SharedPtr map);

    /// 向地图中插入地图点
    void insertMapPoint(MapPointPtr pMp, SharedPtr map);

private:
    std::vector<KeyFramePtr> mvpKeyFrames; ///< 地图中的所有关键帧
    std::vector<MapPointPtr> mvpMapPoints; ///< 地图中的所有地图点
};
} // namespace ORB_SLAM2_ROS2
