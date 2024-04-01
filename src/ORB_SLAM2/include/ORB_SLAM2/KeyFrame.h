#pragma once

#include <list>
#include <memory>

#include "Frame.h"
#include "ORBMatcher.h"

namespace ORB_SLAM2_ROS2 {
class Map;

class KeyFrame : public VirtualFrame {
    friend class ORBMatcher;
    friend class Map;

public:
    typedef std::shared_ptr<KeyFrame> SharedPtr;
    typedef std::shared_ptr<const KeyFrame> ConstSharedPtr;
    typedef std::map<SharedPtr, std::size_t> ConnectedType;
    typedef std::weak_ptr<Map> MapWeakPtr;

    static SharedPtr create(const VirtualFrame &vFrame) {
        SharedPtr pKframe = std::shared_ptr<KeyFrame>(new KeyFrame(vFrame));
        ++mnNextId;
        return pKframe;
    }

    /// 关键帧设置地图
    void setMap(MapWeakPtr pMap) {
        mbIsInMap = true;
        mpMap = pMap;
    }

    bool isBad() const {
        std::unique_lock<std::mutex> lock(mBadMutex);
        return mbIsBad;
    }

    /// 获取关键帧的父节点
    SharedPtr getParent() { return mpParent; }

    /// 获取关键帧的子节点
    std::vector<SharedPtr> getChildren() { return mvpChildren; }

    /// 获取相连关键帧（输入的是权重阈值）
    std::vector<SharedPtr> getConnectedKfs(int th) override;

    /// 更新连接信息
    void updateConnections();

    /// 获取相连关键帧（输入的是相连关键帧的个数）
    std::vector<SharedPtr> getOrderedConnectedKfs(int nNum);

    /// 获取是否在局部地图中
    bool isLocalKf() const { return mbIsLocalKf; }

    /// 设置是否在局部地图中
    void setLocalKf(bool bIsLocalKf) { mbIsLocalKf = bIsLocalKf; }

    ~KeyFrame() = default;

private:
    /**
     * @brief 由普通帧生成关键帧的构造函数
     * @details
     *      1. 使用VirtualFrame的拷贝构造进行必要元素的拷贝
     *      2. 添加地图点到关键帧的观测信息
     * @param vFrame 传入的普通帧
     */
    KeyFrame(const VirtualFrame &vFrame)
        : VirtualFrame(vFrame) {
        mnId = mnNextId;
    }

    static std::size_t mnNextId;                    ///< 下一个关键帧的id
    std::size_t mnId;                               ///< 当前关键帧的id
    bool mbIsInMap = false;                         ///< 是否在地图中
    ConnectedType mmConnectedKfs;                   ///< 产生连接的关键帧
    std::list<KeyFrame::SharedPtr> mlpConnectedKfs; ///< 连接的关键帧(从大到小)
    std::list<std::size_t> mlnConnectedWeights;     ///< 连接权重（从大到小）
    SharedPtr mpParent;                             ///< 生成树的父节点
    std::vector<SharedPtr> mvpChildren;             ///< 生成树的子节点
    MapWeakPtr mpMap;                               ///< 地图
    bool mbIsBad = false;                           ///< 是否是废弃的关键帧
    mutable std::mutex mBadMutex;                   ///< mbIsBad的互斥锁
    bool mbIsLocalKf = false;                       ///< 是否在跟踪线程维护的局部地图中
};
} // namespace ORB_SLAM2_ROS2