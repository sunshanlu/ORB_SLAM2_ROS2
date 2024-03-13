#pragma once

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
    typedef std::multimap<std::size_t, SharedPtr, std::greater<std::size_t>> ConnectedType;
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
        for (std::size_t idx = 0; idx < mvpMapPoints.size(); ++idx) {
            MapPoint::SharedPtr pMp = mvpMapPoints[idx];
            if (pMp)
                pMp->addObservation(this, idx);
        }
    }

    static std::size_t mnNextId;        ///< 下一个关键帧的id
    std::size_t mnId;                   ///< 当前关键帧的id
    bool mbIsInMap = false;             ///< 是否在地图中
    ConnectedType mmConnectedKfs;       ///< 产生连接的关键帧
    SharedPtr mpParent;                 ///< 生成树的父节点
    std::vector<SharedPtr> mvpChildren; ///< 生成树的子节点
    MapWeakPtr mpMap;                   ///< 地图
};
} // namespace ORB_SLAM2_ROS2