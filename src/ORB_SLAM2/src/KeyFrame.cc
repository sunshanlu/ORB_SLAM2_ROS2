#include "ORB_SLAM2/KeyFrame.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 获取与当前帧一阶相连的关键帧（大于给定连接权重阈值的）
 *
 * @param th 给定的连接权重阈值，只统计大于等于这个阈值的相连关键帧
 * @return std::vector<KeyFrame::SharedPtr> 输出一阶相连的关键帧
 */
std::vector<KeyFrame::SharedPtr> KeyFrame::getConnectedKfs(int th) {
    std::vector<SharedPtr> connectedKfs;
    auto wIt = mlnConnectedWeights.begin();
    auto wEnd = mlnConnectedWeights.end();
    auto kfIt = mlpConnectedKfs.begin();
    auto kfEnd = mlpConnectedKfs.end();
    while (wIt != wEnd) {
        if (*wIt >= th)
            connectedKfs.push_back(*kfIt);
        else
            break;
        ++wIt;
        ++kfIt;
    }
    return connectedKfs;
}

/**
 * @brief 更新连接信息，在插入到地图中时调用
 *
 */
void KeyFrame::updateConnections() {
    mlpConnectedKfs.clear();
    mlnConnectedWeights.clear();
    mmConnectedKfs.clear();
    std::map<KeyFrame::SharedPtr, std::size_t> mapConnected;
    for (auto &pMp : mvpMapPoints) {
        if (!pMp || pMp->isBad())
            continue;
        MapPoint::Observations obs = pMp->getObservation();
        for (auto &obsItem : obs)
            ++mapConnected[obsItem.first.lock()];
    }
    std::multimap<std::size_t, KeyFrame::SharedPtr, std::greater<std::size_t>> weightKfs;
    for (auto &item : mapConnected)
        weightKfs.insert(std::make_pair(item.second, item.first));
    for (auto &item : weightKfs) {
        mlpConnectedKfs.push_back(item.second);
        mlnConnectedWeights.push_back(item.first);
    }
    mmConnectedKfs.swap(mapConnected);
}

/**
 * @brief 获取前nNum个与当前帧一阶相连的关键帧
 *
 * @param nNum 输入的要求获取的关键帧数量
 * @return std::vector<KeyFrame::SharedPtr> 输出的满足要求的关键帧
 */
std::vector<KeyFrame::SharedPtr> KeyFrame::getOrderedConnectedKfs(int nNum) {
    std::vector<SharedPtr> connectedKfs;
    if (mlpConnectedKfs.size() < nNum) {
        std::copy(mlpConnectedKfs.begin(), mlpConnectedKfs.end(), std::back_inserter(connectedKfs));
        return connectedKfs;
    }
    auto iter = mlpConnectedKfs.begin();
    for (int i = 0; i < nNum; ++i) {
        connectedKfs.push_back(*iter);
        ++iter;
    }
    return connectedKfs;
}

std::size_t KeyFrame::mnNextId;
} // namespace ORB_SLAM2_ROS2