#pragma once

#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
namespace ORB_SLAM2_ROS2 {
class KeyFrame;
class Map;

class MapPoint {
    friend class Map;

public:
    typedef std::shared_ptr<MapPoint> SharedPtr;
    typedef std::map<KeyFrame *, std::size_t> Observations;
    typedef std::weak_ptr<Map> MapWeakPtr;

    static MapPoint::SharedPtr create(cv::Mat mP3d) {
        MapPoint::SharedPtr pMpoint(new MapPoint(mP3d));
        ++mnNextId;
        return pMpoint;
    }

    /// 地图点添加观测信息
    void addObservation(KeyFrame *pkf, std::size_t featId);

    /// 地图点设置地图
    void setMap(MapWeakPtr pMap) {
        mbIsInMap = true;
        mpMap = pMap;
    }

    /// 判断地图点是否是bad
    bool isBad() const;

private:
    MapPoint(cv::Mat mP3d)
        : mPoint3d(std::move(mP3d)) {
        mId = mnNextId;
    }

    static unsigned int mnNextId; ///< 下一个地图点的id
    unsigned int mId;             ///< this的地图点id
    cv::Mat mPoint3d;             ///< this的地图点3d坐标（世界坐标系下）
    bool mbIsInMap = false;       ///< 是否在地图中
    Observations mObs;            ///< 地图点的观测
    MapWeakPtr mpMap;             ///< 地图
    bool mbIsBad = false;         ///< 地图点是否是bad
    std::mutex mBadMutex;         ///< 维护mbIsBad的互斥锁
};
} // namespace ORB_SLAM2_ROS2