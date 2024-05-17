#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 获取与当前帧一阶相连的关键帧（大于给定连接权重阈值的）
 *
 * @param th 给定的连接权重阈值，只统计大于等于这个阈值的相连关键帧
 * @return std::vector<KeyFrame::SharedPtr> 输出一阶相连的关键帧
 */
std::vector<KeyFrame::SharedPtr> KeyFrame::getConnectedKfs(int th) {
    decltype(mlnConnectedWeights) lnConnectedWeights;
    decltype(mlpConnectedKfs) lpConnectedKfs;
    {
        std::unique_lock<std::mutex> lock(mConnectedMutex);
        lnConnectedWeights = mlnConnectedWeights;
        lpConnectedKfs = mlpConnectedKfs;
    }
    std::vector<SharedPtr> connectedKfs;
    auto wIt = lnConnectedWeights.begin();
    auto wEnd = lnConnectedWeights.end();
    auto kfIt = lpConnectedKfs.begin();
    auto kfEnd = lpConnectedKfs.end();
    while (wIt != wEnd) {
        if (*wIt >= th) {
            SharedPtr pkf = kfIt->lock();
            if (pkf && !pkf->isBad())
                connectedKfs.push_back(pkf);
        } else
            break;
        ++wIt;
        ++kfIt;
    }
    return connectedKfs;
}

/**
 * @brief 更新连接信息，在插入到地图中时调用
 * @details
 *      1. 通过关键帧的地图点的观测信息，进行连接权重统计
 *      2. 将权重大于等于15的部分进行统计（视为产生了连接关系）
 *          1. 如何没有大于等于15的连接关系，将最大连接保留下来
 *      3. 按照共视程度，从大到小进行排列
 *      4. 初始化当前共视程度最高的关键帧为父关键帧，当前关键帧为子关键帧
 */
KeyFrame::SharedPtr KeyFrame::updateConnections() {
    {
        std::unique_lock<std::mutex> lock(mConnectedMutex);
        mlpConnectedKfs.clear();
        mlnConnectedWeights.clear();
        mmConnectedKfs.clear();
    }
    std::map<KeyFrame::SharedPtr, std::size_t> mapConnected;
    auto vpMapPoints = getMapPoints();
    for (auto &pMp : vpMapPoints) {
        if (!pMp || pMp->isBad())
            continue;
        MapPoint::Observations obs = pMp->getObservation();
        for (auto &obsItem : obs) {
            auto pkf = obsItem.first.lock();
            if (!pkf || this == pkf.get() || pkf->isBad())
                continue;
            if (pkf->getID() > mnId)
                continue;
            ++mapConnected[pkf];
        }
    }

    std::size_t maxWeight = 0;
    SharedPtr pBestPkf = nullptr;

    std::multimap<std::size_t, KeyFrame::SharedPtr, std::greater<std::size_t>> weightKfs;
    {
        std::unique_lock<std::mutex> lock(mConnectedMutex);

        for (auto &item : mapConnected) {
            if (item.second > maxWeight) {
                maxWeight = item.second;
                pBestPkf = item.first;
            }
            if (item.second < 15)
                continue;
            weightKfs.insert(std::make_pair(item.second, item.first));
            mmConnectedKfs.insert(std::make_pair(item.first, item.second));
        }
        if (weightKfs.empty() && pBestPkf && !pBestPkf->isBad()) {
            weightKfs.insert(std::make_pair(maxWeight, pBestPkf));
            mmConnectedKfs.insert(std::make_pair(pBestPkf, maxWeight));
        }
        for (auto &item : weightKfs) {
            mlpConnectedKfs.push_back(item.second);
            mlnConnectedWeights.push_back(item.first);
        }
    }

    return pBestPkf;
}

/**
 * @brief 更新连接关系（更新父子关系）
 * @details
 *      1. 在局部建图线程中，更新的父关键帧的id一定在前（生成树一定不会闭环）
 *      2. 生成树不会闭环，保证父关键帧的id一定是小于子关键帧id的
 * @param child 输入的待更新连接权重的关键帧
 */
void KeyFrame::updateConnections(SharedPtr child) {
    // std::cout << "进入更新连接函数" << std::endl;
    SharedPtr parent = child->updateConnections();
    // std::cout << "更新连接部分成功" << std::endl;
    if (!parent || parent->getID() > child->getID())
        return;
    if (child->isParent()) {
        SharedPtr originParent = child->getParent().lock();
        if (originParent && !originParent->isBad())
            originParent->eraseChild(child);
    }
    parent->addChild(child);
    child->setParent(parent);
}

/**
 * @brief 获取前nNum个与当前帧一阶相连的关键帧
 *
 * @param nNum 输入的要求获取的关键帧数量
 * @return std::vector<KeyFrame::SharedPtr> 输出的满足要求的关键帧
 */
std::vector<KeyFrame::SharedPtr> KeyFrame::getOrderedConnectedKfs(int nNum) {
    decltype(mlpConnectedKfs) lpConnectedKfs;
    {
        std::unique_lock<std::mutex> lock(mConnectedMutex);
        lpConnectedKfs = mlpConnectedKfs;
    }
    std::vector<SharedPtr> connectedKfs;
    if (lpConnectedKfs.size() < nNum) {
        for (auto iter = lpConnectedKfs.begin(); iter != lpConnectedKfs.end(); ++iter) {
            SharedPtr pkf = iter->lock();
            if (pkf && !pkf->isBad())
                connectedKfs.push_back(pkf);
        }
        return connectedKfs;
    }
    auto iter = lpConnectedKfs.begin();
    int n = 0;
    for (auto &pkfWeak : lpConnectedKfs) {
        SharedPtr pkf = iter->lock();
        if (pkf && pkf->isBad()) {
            connectedKfs.push_back(pkf);
            ++n;
        }
        if (n >= nNum)
            break;
    }
    return connectedKfs;
}

/**
 * @brief 删除指定的共视关系
 * @details
 *      1. mmConnectedKfs的删除
 *      2. mlpConnectedKfs的删除
 *      3. mlnConnectedWeights的删除
 * @param pkf 输入的要删除的连接关系
 */
void KeyFrame::eraseConnection(SharedPtr pkf) {
    std::unique_lock<std::mutex> lock(mConnectedMutex);
    auto iter = mmConnectedKfs.find(pkf);
    if (iter != mmConnectedKfs.end()) {
        mmConnectedKfs.erase(iter);
        auto nIter = mlnConnectedWeights.begin();
        for (auto pIter = mlpConnectedKfs.begin(); pIter != mlpConnectedKfs.end(); ++pIter) {
            if (pIter->lock() == pkf) {
                mlpConnectedKfs.erase(pIter);
                mlnConnectedWeights.erase(nIter);
                break;
            }
            ++nIter;
        }
    }
}

/// KeyFrame的静态变量
std::size_t KeyFrame::mnNextId;
KeyFrame::WeakCompareFunc KeyFrame::weakCompare = [](KeyFrame::WeakPtr p1, KeyFrame::WeakPtr p2) {
    static long idx = -1;
    int p1Id = 0;
    int p2Id = 0;
    auto sharedP1 = p1.lock();
    auto sharedP2 = p2.lock();
    if (!sharedP1 || sharedP1->isBad())
        p1Id = idx--;
    else
        p1Id = sharedP1->getID();
    if (!sharedP2 || sharedP2->isBad())
        p2Id = idx--;
    else
        p2Id = sharedP2->getID();
    return p1Id > p2Id;
};
} // namespace ORB_SLAM2_ROS2