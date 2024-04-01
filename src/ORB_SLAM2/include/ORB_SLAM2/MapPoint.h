#pragma once

#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
namespace ORB_SLAM2_ROS2 {
class KeyFrame;
class Map;
class VirtualFrame;

class MapPoint {
    friend class Map;

public:
    typedef std::shared_ptr<MapPoint> SharedPtr;
    typedef std::weak_ptr<Map> MapWeakPtr;
    typedef std::weak_ptr<KeyFrame> KeyFrameWeakPtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<VirtualFrame> VirtualFramePtr;
    typedef std::vector<std::pair<KeyFrameWeakPtr, std::size_t>> Observations;

    /// 由关键帧构造
    static MapPoint::SharedPtr create(cv::Mat mP3dW, KeyFramePtr pKf, std::size_t nObs);

    /// 由普通帧构造
    static MapPoint::SharedPtr create(cv::Mat mP3dW);

    /// 地图点添加观测信息
    void addObservation(KeyFrameWeakPtr pkf, std::size_t featId);

    /// 获取观测信息
    Observations getObservation() { return mObs; }
    const Observations getObservation() const { return mObs; }

    /// 地图点设置地图
    void setMap(MapWeakPtr pMap) {
        mbIsInMap = true;
        mpMap = pMap;
    }

    /// 判断地图点是否是bad
    bool isBad() const;

    /// 判断地图点是否在map中
    bool isInMap() const { return mbIsInMap; };

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

    /// 获取是否在局部地图中
    bool isLocalMp() const { return mbIsLocalMp; }

    /// 设置是否在局部地图中
    void setLocalMp(bool bIsLocalMp) { mbIsLocalMp = bIsLocalMp; }

    /// 预测金字塔层级
    int predictLevel(float distance) const;

    /// 跟踪线程初始化专用函数（由普通地图点添加属性到特殊地图点上）
    void addAttriInit(KeyFramePtr pRefKf, std::size_t nFeatId);

    /// 判断某一位姿是否在共视范围内
    bool isInVision(VirtualFramePtr pFrame, float &vecDistance, cv::Point2f &uv, float &cosTheta);

    /// 获取地图点的代表描述子
    cv::Mat getDesc() const { return mDescriptor.clone(); }

    /// 添加跟踪的匹配信息
    void addMatchInTrack() { ++mnMatchesInTrack; }

    /// 添加跟踪的优化内点信息
    void addInlierInTrack() { ++mnInliersInTrack; }

    /// 返回地图点在跟踪过程中的重要程度
    float scoreInTrack() { return (float)mnMatchesInTrack / mnInliersInTrack; }

private:
    /// 用于关键帧构造的地图点
    MapPoint(cv::Mat mP3d, KeyFramePtr pRefKf, std::size_t nObs);

    /// 用于普通帧构造的临时地图点
    MapPoint(cv::Mat mP3d);

    static unsigned int mnNextId; ///< 下一个地图点的id
    unsigned int mId;             ///< this的地图点id
    cv::Mat mPoint3d;             ///< this的地图点3d坐标（世界坐标系下）
    bool mbIsInMap = false;       ///< 是否在地图中
    Observations mObs;            ///< 地图点的观测
    MapWeakPtr mpMap;             ///< 地图
    bool mbIsBad = false;         ///< 地图点是否是bad
    mutable std::mutex mBadMutex; ///< 维护mbIsBad的互斥锁
    mutable std::mutex mPosMutex; ///< 维护mPoint3d的互斥锁
    cv::Mat mDescriptor;          ///< 地图点的代表描述子
    cv::Mat mViewDirection;       ///< 地图点的观测方向（光心->地图点）
    float mnMaxDistance;          ///< 最大匹配距离
    float mnMinDistance;          ///< 最小匹配距离
    KeyFrameWeakPtr mpRefKf;      ///< 地图点的参考关键帧
    int mnMatchesInTrack = 0;     ///< 在跟踪过程中，被匹配成功的
    int mnInliersInTrack = 0;     ///< 在跟踪过程中，经过优化后还是内点的

public:
    bool mbIsLocalMp = false; ///< 是否在跟踪线程的局部地图中
};
} // namespace ORB_SLAM2_ROS2