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
    typedef std::weak_ptr<KeyFrame> WeakPtr;
    typedef std::shared_ptr<KeyFrame> SharedPtr;
    typedef std::function<bool(WeakPtr, WeakPtr)> WeakCompareFunc;
    typedef std::map<WeakPtr, std::size_t, WeakCompareFunc> ConnectedType;
    typedef std::weak_ptr<Map> MapWeakPtr;

    static SharedPtr create(const VirtualFrame &vFrame) {
        SharedPtr pKframe = std::shared_ptr<KeyFrame>(new KeyFrame(vFrame));
        ++mnNextId;
        return pKframe;
    }

    /// 获取关键帧ID
    std::size_t getID() const { return mnId; }

    /// 关键帧设置地图
    void setMap(MapWeakPtr pMap) {
        mbIsInMap = true;
        mpMap = pMap;
    }

    /// 设置map为null
    void setMapNull() {
        mpMap.reset();
        mbIsInMap = false;
    }

    bool isBad() const {
        std::unique_lock<std::mutex> lock(mBadMutex);
        return mbIsBad;
    }

    /// 获取关键帧的父节点
    WeakPtr getParent() {
        std::unique_lock<std::mutex> lock(mTreeMutex);
        return mpParent;
    }

    /// 父节点是否存在
    bool isParent() {
        std::unique_lock<std::mutex> lock(mTreeMutex);
        auto pParent = mpParent.lock();
        return pParent && !pParent->isBad();
    }

    /// 设置父节点
    void setParent(SharedPtr pkf) {
        std::unique_lock<std::mutex> lock(mTreeMutex);
        mpParent = pkf;
    }

    /// 获取关键帧的子节点
    std::set<WeakPtr, WeakCompareFunc> getChildren() {
        std::unique_lock<std::mutex> lock(mTreeMutex);
        return mspChildren;
    }

    /// 添加子关键帧
    void addChild(SharedPtr child) {
        std::unique_lock<std::mutex> lock(mTreeMutex);
        mspChildren.insert(child);
    }

    /// 设置bad flag
    void setBad() {
        std::unique_lock<std::mutex> lock(mBadMutex);
        mbIsBad = true;
    }

    /// 获取关键帧是否不能被删除
    bool isNotErased() const { return mbNotErase; }

    /// 获取相连关键帧（输入的是权重阈值）
    std::vector<SharedPtr> getConnectedKfs(int th) override;

    /// 获取相连关键帧（输入的是相连关键帧的个数）
    std::vector<SharedPtr> getOrderedConnectedKfs(int nNum);

    /// 获取所有一阶相连关键帧和权重
    ConnectedType getAllConnected() {
        std::unique_lock<std::mutex> lock(mConnectedMutex);
        return mmConnectedKfs;
    }

    /// 获取是否在局部地图中
    bool isLocalKf() const { return mbIsLocalKf; }

    /// 设置是否在局部地图中
    void setLocalKf(bool bIsLocalKf) { mbIsLocalKf = bIsLocalKf; }

    ~KeyFrame() = default;

    /// 删除某个子关键帧
    void eraseChild(SharedPtr pkf) {
        std::unique_lock<std::mutex> lock(mTreeMutex);
        auto iter = mspChildren.find(pkf);
        if (iter == mspChildren.end())
            return;
        mspChildren.erase(iter);
    }

    /// 更新连接关系（更新父子关系）
    static void updateConnections(SharedPtr child);

    /// 删除指定关键帧的共视关系
    void eraseConnection(SharedPtr pkf);

private:
    /// 更新连接信息
    SharedPtr updateConnections();

    /**
     * @brief 由普通帧生成关键帧的构造函数
     * @details
     *      1. 使用VirtualFrame的拷贝构造进行必要元素的拷贝
     *      2. 添加地图点到关键帧的观测信息
     * @param vFrame 传入的普通帧
     */
    KeyFrame(const VirtualFrame &vFrame)
        : VirtualFrame(vFrame)
        , mspChildren(weakCompare)
        , mmConnectedKfs(weakCompare) {
        mnId = mnNextId;
    }

    static std::size_t mnNextId;                    ///< 下一个关键帧的id
    std::size_t mnId;                               ///< 当前关键帧的id
    bool mbIsInMap = false;                         ///< 是否在地图中
    ConnectedType mmConnectedKfs;                   ///< 产生连接的关键帧
    std::list<WeakPtr> mlpConnectedKfs;             ///< 连接的关键帧(从大到小)
    std::list<std::size_t> mlnConnectedWeights;     ///< 连接权重（从大到小）
    WeakPtr mpParent;                               ///< 生成树的父节点
    std::set<WeakPtr, WeakCompareFunc> mspChildren; ///< 生成树的子节点
    MapWeakPtr mpMap;                               ///< 地图
    bool mbIsBad = false;                           ///< 是否是废弃的关键帧
    mutable std::mutex mBadMutex;                   ///< mbIsBad的互斥锁
    mutable std::mutex mConnectedMutex;             ///< 维护共视关系的互斥锁
    mutable std::mutex mTreeMutex;                  ///< 维护生成树关系的互斥锁
    bool mbIsLocalKf = false;                       ///< 是否在跟踪线程维护的局部地图中
    bool mbNotErase = false;                        ///< 当参与回环闭合时，不能删除

public:
    static WeakCompareFunc weakCompare; ///< 关键帧弱指针比较函数
};

} // namespace ORB_SLAM2_ROS2