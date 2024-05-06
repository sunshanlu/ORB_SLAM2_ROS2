#pragma once

#include <memory>
#include <queue>
#include <set>

namespace ORB_SLAM2_ROS2 {

class KeyFrame;
class KeyFrameDB;

class LocalMapping {
public:
    typedef std::shared_ptr<LocalMapping> SharedPtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<KeyFrameDB> KeyFrameDBPtr;
    typedef std::pair<std::set<KeyFramePtr>, int> ConsistGroup;

    /// 检测回环是否产生
    bool detectLoop();

    /// 计算Sim3相似性变换矩阵
    bool computeSim3();

private:
    std::queue<KeyFramePtr> mqKeyFrames;       ///< 待回环检测的关键帧队列
    KeyFramePtr mpCurrKeyFrame;                ///< 当前待回环检测的关键帧
    KeyFrameDBPtr mpKeyFrameDB;                ///< 关键帧数据库
    std::vector<ConsistGroup> mvConsistGroups; ///< 连续性组链
    std::vector<KeyFramePtr> mvEnoughKfs;      ///< 达到连续性条件的候选关键帧
};

} // namespace ORB_SLAM2_ROS2