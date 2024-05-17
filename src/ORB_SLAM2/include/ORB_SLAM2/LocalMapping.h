#pragma once
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

#include "ORB_SLAM2/KeyFrame.h"

namespace ORB_SLAM2_ROS2 {

class KeyFrame;
class MapPoint;
class Map;
class KeyFrameDB;
class LoopClosing;

class LocalMapping {
public:
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<MapPoint> MapPointPtr;
    typedef std::unordered_map<std::size_t, MapPointPtr> UnprocessMps;
    typedef std::shared_ptr<Map> MapPtr;
    typedef std::weak_ptr<KeyFrame> KeyFrameWeak;
    typedef std::map<MapPointPtr, std::vector<std::size_t>> MapPointDB;
    typedef std::shared_ptr<KeyFrameDB> KeyFrameDBPtr;
    typedef std::shared_ptr<LoopClosing> LoopClosingPtr;
    /// weightDB的模版参数分别代表共视权重、子关键帧、候选父关键帧
    typedef std::multimap<std::size_t, std::pair<KeyFramePtr, KeyFramePtr>, std::greater<std::size_t>> WeightDB;

    /// 局部建图线程的构造函数
    LocalMapping(MapPtr pMap);

    /// 局部建图线程入口函数
    void run();

    /// 一次后端循环处理的过程
    void runOnce();

    /// 删除冗余地图点
    void cullingMapPoints();

    /// 处理新关键帧
    void processNewKeyFrame(KeyFramePtr pkf);

    /// 插入关键帧
    void insertKeyFrame(KeyFramePtr pkf);

    /// 获取需要处理的关键帧
    KeyFramePtr getNewKeyFrame();

    /// 生成新的地图点（针对的是没有产生匹配的ORB特征点，绝对的新增）
    void createNewMapPoints();

    /// 地图点的融合
    void fuseMapPoints();

    /// 删除冗余关键帧
    void cullingKeyFrames();

    /// 设置闭环检测器
    void setLoopClosing(LoopClosingPtr pLoopCloser) { mpLoopCloser = pLoopCloser; }

    /// 三角化
    cv::Mat triangulate(KeyFramePtr pkf1, KeyFramePtr pkf2, const cv::KeyPoint &kp1, const cv::KeyPoint &kp2);

    /// 根据位姿和2d点，计算夹角Theta的cos值
    float computeCosTheta(cv::Mat R1w, cv::Mat R2w, const cv::Point2f &pt1, const cv::Point2f &pt2);

    /// 使用最小生成树的方法，寻找子关键帧的父关键帧
    void findParent(std::set<KeyFrameWeak, KeyFrame::WeakCompareFunc> &spChildren,
                    std::vector<KeyFramePtr> &vpCandidates);

    /// 设置是否接受关键帧
    void setAccpetKF(const bool &flag) {
        std::unique_lock<std::mutex> mMutexAcceptKF;
        mbAcceptKF = flag;
    }

    bool getAcceptKF() {
        std::unique_lock<std::mutex> mMutexAcceptKF;
        return mbAcceptKF;
    }

    /// 是否处理完成关键帧队列
    bool isCompleted() {
        std::unique_lock<std::mutex> lock(mMutexQueue);
        return mqpKeyFrames.empty();
    }

    /// 返回待处理关键帧的数目
    std::size_t getKFNum() {
        std::unique_lock<std::mutex> lock(mMutexQueue);
        return mqpKeyFrames.size();
    }

    /// 设置BA中断
    void setAbortBA(const bool &flag) {
        std::unique_lock<std::mutex> lock(mMutexAbortBA);
        mbAbortBA = flag;
    }

    /// 将当前关键帧插入关键帧数据库中
    void addKF2DB(KeyFrameDBPtr pkfDB);

    /// 请求局部建图线程的停止
    void requestStop() {
        {
            std::unique_lock<std::mutex> lock(mMutexRequestStop);
            mbRequestStop = true;
        }
        setAccpetKF(false);
        setAbortBA(true);
    }

    void start() {
        {
            std::unique_lock<std::mutex> lock(mMutexRequestStop);
            mbRequestStop = false;
        }
        setStop(false);
        setAccpetKF(true);
    }

    /// 是否有外部停止命令
    bool isRequestStop() {
        std::unique_lock<std::mutex> lock(mMutexRequestStop);
        return mbRequestStop;
    }

    /// 判断已经停止，设置停止标识符
    void setStop(bool flag) {
        std::unique_lock<std::mutex> lock(mMutexStoped);
        mbStoped = flag;
    }

    /// 判断局部建图线程是否已经停止
    bool isStop() {
        std::unique_lock<std::mutex> lock(mMutexStoped);
        return mbStoped;
    }

private:
    /// 创建地图点数据库
    void createMpsDB(std::vector<KeyFramePtr> &vpTargetKfs, MapPointDB &mapPointDB);

    /// 判断关键帧是否冗余
    bool judgeKeyFrame(KeyFramePtr &pkf, const MapPointDB &mapPointDB);

    /// 删除给定关键帧
    void deleteKeyFrame(KeyFramePtr &pkf);

    KeyFramePtr mpCurrKeyFrame;           ///< 当前关键帧
    std::queue<KeyFramePtr> mqpKeyFrames; ///< 待处理的关键帧队列
    std::list<MapPointPtr> mlpAddedMPs;   ///< 新生成的地图点
    MapPtr mpMap;                         ///< 地图
    UnprocessMps mmUnprocessMps;          ///< 未经过处理的地图点(本身投影产生的地图点)
    std::set<KeyFramePtr> mspToBeErased;  ///< 参与回环闭合，没有删除的关键帧
    bool mbAcceptKF = true;               ///< 是否可接收关键帧
    bool mbAbortBA = false;               ///< 是否需要阻止BA优化
    mutable std::mutex mMutexQueue;       ///< 待处理关键帧队列锁
    mutable std::mutex mMutexAcceptKF;    ///< 维护mbAcceptKF的互斥锁
    mutable std::mutex mMutexAbortBA;     ///< 维护mbAbortBA的互斥锁
    mutable std::mutex mMutexRequestStop; ///< 维护mbRequestStop的互斥锁
    mutable std::mutex mMutexStoped;      ///< 维护mbStoped的互斥锁
    bool mbRequestStop = false;           ///< 外部有请求停止的命令
    bool mbStoped = false;                ///< 局部建图线程停止的标识
    LoopClosingPtr mpLoopCloser;          ///< 闭环检测器对象
};

} // namespace ORB_SLAM2_ROS2
