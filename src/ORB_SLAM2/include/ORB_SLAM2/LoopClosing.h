#pragma once

#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_ROS2 {

class KeyFrame;
class KeyFrameDB;
class Sim3Ret;
class MapPoint;
class Map;
class LocalMapping;
class Tracking;

class LoopClosing {
public:
    typedef std::shared_ptr<LoopClosing> SharedPtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<MapPoint> MapPointPtr;
    typedef std::shared_ptr<KeyFrameDB> KeyFrameDBPtr;
    typedef std::pair<std::set<KeyFramePtr>, int> ConsistGroup;
    typedef std::map<KeyFramePtr, Sim3Ret> KeyFrameAndSim3;
    typedef std::shared_ptr<Map> MapPtr;
    typedef std::shared_ptr<LocalMapping> LocalMappingPtr;
    typedef std::shared_ptr<Tracking> TrackingPtr;

    LoopClosing(KeyFrameDBPtr pKeyFrameDB, MapPtr pMap, LocalMappingPtr pLocalMapping, TrackingPtr pTracking);

    /// 检测回环是否产生
    bool detectLoop();

    /// 计算Sim3相似性变换矩阵
    bool computeSim3(Sim3Ret &g2oScm, Sim3Ret &g2oScw, KeyFramePtr &pLoopKf);

    /// 回环矫正
    void correctLoop(const Sim3Ret &g2oScw, const KeyFramePtr &pLoopKf);

    /// 回环闭合线程入口函数
    void run();

    /// 循环运行一次
    void runOnce();

    /// 处理新关键帧
    bool processNewKeyFrame();

    /// 插入新关键帧
    void insertKeyFrame(KeyFramePtr pkf);

    /// 矫正完成关键帧，运行全局BA
    void runGlobalBA();

    /// 释放回环闭合线程的资源
    void release();

    /// 请求回环闭合线程停止
    void requestStop() {
        std::unique_lock<std::mutex> lock(mMutexResStop);
        mbResquestStop = true;
        mbStopGBA = true;
    }

    /// 回环闭合线程是否被请求停止
    bool isRequestStop() const {
        std::unique_lock<std::mutex> lock(mMutexResStop);
        return mbResquestStop;
    }

    /// 回环闭合线程是否停止
    bool isStop() const {
        std::unique_lock<std::mutex> lock(mMutexStop);
        return mbStop;
    }

    /// 停止回环闭合线程
    void stop() {
        std::unique_lock<std::mutex> lock(mMutexStop);
        mbStop = true;
    }

private:
    /// 判断两个组之间是否连续
    bool isConsistBetween(const std::set<KeyFramePtr> &groupDB, const std::set<KeyFramePtr> &groupFind);

    std::queue<KeyFramePtr> mqKeyFrames;       ///< 待回环检测的关键帧队列
    mutable std::mutex mMutexQueue;            ///< mqKeyFrames的队列锁
    KeyFramePtr mpCurrKeyFrame;                ///< 当前待回环检测的关键帧
    KeyFrameDBPtr mpKeyFrameDB;                ///< 关键帧数据库
    std::vector<ConsistGroup> mvConsistGroups; ///< 连续性组链
    std::vector<KeyFramePtr> mvEnoughKfs;      ///< 达到连续性条件的候选关键帧
    std::vector<MapPointPtr> mvLoopGroupMps;   ///< 回环闭合关键帧组的地图点
    std::vector<MapPointPtr> mvMatchedMps;     ///< msLoopGroupMps中匹配到当前帧的地图点
    MapPtr mpMap;                              ///< 地图
    LocalMappingPtr mpLocalMapper;             ///< 局部建图对象
    TrackingPtr mpTracker;                     ///< 跟踪对象
    std::size_t mnLastLoopId;                  ///< 上一次回环闭合的关键帧id
    bool mbStopGBA;                            ///< 默认不停止全局BA
    std::thread *mpGlobalBAThread = nullptr;   ///< 全局BA线程
    bool mbStop;                               ///< 回环闭合线程是否停止
    bool mbResquestStop;                       ///< 回环闭合线程是否请求停止
    mutable std::mutex mMutexStop;             ///< 维护mbStop的互斥锁
    mutable std::mutex mMutexResStop;          ///< 维护mbResquestStop的互斥锁
};

} // namespace ORB_SLAM2_ROS2