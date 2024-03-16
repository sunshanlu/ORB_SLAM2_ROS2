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
    typedef std::weak_ptr<Map> MapWeakPtr;
    typedef std::weak_ptr<KeyFrame> KeyFrameWeakPtr;
    typedef std::vector<std::pair<KeyFrameWeakPtr, std::size_t>> Observations;

    static MapPoint::SharedPtr create(cv::Mat mP3d) {
        MapPoint::SharedPtr pMpoint(new MapPoint(mP3d));
        ++mnNextId;
        return pMpoint;
    }

    /// 地图点添加观测信息
    void addObservation(KeyFrameWeakPtr pkf, std::size_t featId);

    /// 地图点设置地图
    void setMap(MapWeakPtr pMap) {
        mbIsInMap = true;
        mpMap = pMap;
    }

    /// 判断地图点是否是bad
    bool isBad() const;

    /// 返回地图点位置
    cv::Mat getPos() const {
        cv::Mat pos;
        std::unique_lock<std::mutex> lock(mPosMutex);
        mPoint3d.copyTo(pos);
        return pos;
    }

    /// 设置地图点位置
    void setPos(cv::Mat point3d) {
        std::unique_lock<std::mutex> lock(mPosMutex);
        point3d.copyTo(mPoint3d);
    }

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
    mutable std::mutex mBadMutex; ///< 维护mbIsBad的互斥锁
    mutable std::mutex mPosMutex; ///< 维护mPoint3d的互斥锁
};
} // namespace ORB_SLAM2_ROS2