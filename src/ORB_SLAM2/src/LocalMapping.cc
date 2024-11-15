#include <unordered_set>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/LocalMapping.h"
#include "ORB_SLAM2/LoopClosing.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Optimizer.h"

using namespace std::chrono_literals;

namespace ORB_SLAM2_ROS2 {

/// 局部建图线程的构造函数
LocalMapping::LocalMapping(Map::SharedPtr pMap)
    : mpMap(pMap) {}

/**
 * @brief 获取需要处理的关键帧
 *
 * @return KeyFrame::SharedPtr 输出关键帧的共享指针
 */
KeyFrame::SharedPtr LocalMapping::getNewKeyFrame() {
    std::unique_lock<std::mutex> lock(mMutexQueue);
    if (mqpKeyFrames.empty()) {
        lock.unlock();
        std::this_thread::sleep_for(10ms);
        return nullptr;
    } else {
        KeyFrame::SharedPtr pKf = mqpKeyFrames.front();
        mqpKeyFrames.pop();
        lock.unlock();
        return pKf;
    }
}

/**
 * @brief 局部建图线程的入口函数
 *
 */
void LocalMapping::run() {
    while (!isFinished() && rclcpp::ok()) {
        if (!isStop()) {
            runOnce();
            std::this_thread::sleep_for(3ms);
        } else {
            std::this_thread::sleep_for(10ms);
        }
    }
}

/**
 * @brief 局部建图线程的单次运行函数
 * @details
 *      1. 设置不接收关键帧
 *      2. 处理新关键帧
 *      3. 剔除不合格地图点
 *      4. 生成新的地图点
 *      5. 关键帧队列是否处理完毕
 *          (1) 如果关键帧队列处理完毕，搜索融合相邻关键帧地图点
 *          (2) 判断地图点中，关键帧数据是否超过2，进行局部BA优化
 *          (3) 剔除冗余关键帧
 *      6. 设置可接收关键帧
 */
void LocalMapping::runOnce() {
    setAccpetKF(false);
    auto pkf = getNewKeyFrame();
    if (pkf) {
        processNewKeyFrame(pkf);
        cullingMapPoints();
        createNewMapPoints();
        if (isCompleted()) {
            fuseMapPoints();
        }
        if (isCompleted() && !isRequestStop()) {
            setAbortBA(false);
            if (mpMap->keyFramesInMap() > 2)
                Optimizer::OptimizeLocalMap(mpCurrKeyFrame, mbAbortBA);

            cullingKeyFrames();
        }
        if (mpLoopCloser)
            mpLoopCloser->insertKeyFrame(mpCurrKeyFrame);
    } else if (!pkf && isCompleted() && isRequestStop()) {
        setStop(true);
    }
    setAccpetKF(true);
}

/**
 * @brief 处理新关键帧
 * @details
 *      1. 词袋向量的计算
 *      2. 更新有效地图点的平均观测方向、观测距离和最佳描述子
 *      3. 如果地图点是新增地图点，添加观测属性，并放入新增地图点集合中
 *      4. 更新当前关键帧和共视关键帧的连接关系
 *      5. 将该关键帧插入到地图中
 * @param pKf 输入的待处理的关键帧
 */
void LocalMapping::processNewKeyFrame(KeyFramePtr pKf) {
    pKf->computeBow();
    mmUnprocessMps.clear();
    auto mapPoints = pKf->getMapPoints();
    for (std::size_t idx = 0; idx < mapPoints.size(); ++idx) {
        MapPoint::SharedPtr pMp = mapPoints[idx];
        /// MapPoint::isInMap()不用加锁的原因在于调用时机总是在单一线程中进行的
        if (pMp && !pMp->isBad()) {
            if (pMp->isInMap()) {
                pMp->addObservation(pKf, idx);
                pMp->updateDescriptor();
                pMp->updateNormalAndDepth();
            } else {
                mmUnprocessMps.insert(std::make_pair(idx, pMp));
                pKf->setMapPoint(idx, nullptr);
            }
        }
    }
    KeyFrame::updateConnections(pKf);
    mpMap->insertKeyFrame(pKf, mpMap);
    mpCurrKeyFrame = pKf;
}

/**
 * @brief 新增地图点
 * @details
 *      1. 处理关键帧之间产生的地图点
 *          1. 10个共视程度最佳的关键帧
 *          2. 进行词袋的快速匹配（关键帧和关键帧之间）
 *              2.1 匹配前提是基线足够大(基线大于相机基线)
 *              2.2 匹配之后，使用极线约束来抑制离群点
 *          3. 使用SVD进行三角化
 *          4. 剔除不合格地图点
 *              1. 三维点应在相机前方
 *              2. 重投影误差应小于阈值
 *              3. 确保尺度连续性（金字塔缩放因子比例应该和距离的比例相差不大）
 *      2. 处理本身产生的地图点
 */
void LocalMapping::createNewMapPoints() {
    std::map<KeyFrame::SharedPtr, std::vector<cv::DMatch>> matches;
    auto vpConnectedKfs = mpCurrKeyFrame->getOrderedConnectedKfs(10);
    int nMatches = 0;
    for (std::size_t idx = 0; idx < vpConnectedKfs.size(); ++idx) {
        KeyFrame::SharedPtr pKf = vpConnectedKfs[idx];
        if (!pKf || pKf->isBad())
            continue;
        cv::Mat diff = mpCurrKeyFrame->getFrameCenter() - pKf->getFrameCenter();
        float baseline = cv::norm(diff);
        if (baseline < Camera::mfBl)
            continue;
        std::vector<cv::DMatch> vMatches;
        ORBMatcher matcher(0.6f, false);
        nMatches += matcher.searchForTriangulation(mpCurrKeyFrame, pKf, vMatches);
        matches.insert(std::make_pair(pKf, vMatches));
    }
    const auto &kps1 = mpCurrKeyFrame->getLeftKeyPoints();
    cv::Mat R1w, t1w;
    mpCurrKeyFrame->getPose(R1w, t1w);
    const auto &depths1 = mpCurrKeyFrame->getDepth();
    const auto &rightUs1 = mpCurrKeyFrame->getRightU();

    /// 处理关键帧之间产生的新地图点
    for (auto match : matches) {
        const auto &kps2 = match.first->getLeftKeyPoints();
        const auto &depths2 = match.first->getDepth();
        const auto &rightUs2 = match.first->getRightU();
        cv::Mat R2w, _;
        match.first->getPose(R2w, _);

        for (auto &m : match.second) {
            auto pCurrMp = mpCurrKeyFrame->getMapPoint(m.queryIdx);
            if (pCurrMp && !pCurrMp->isBad() && pCurrMp->isInMap())
                continue;
            const auto &kp1 = kps1[m.queryIdx];
            const auto &kp2 = kps2[m.trainIdx];
            float cosTheta0 = 1, cosTheta1 = 1, cosTheta2 = 1;
            cosTheta0 = computeCosTheta(R1w, R2w, kp1.pt, kp2.pt);
            bool bStereo1 = false, bStereo2 = false;
            if (depths1[m.queryIdx] > 0) {
                bStereo1 = true;
                cv::Point2f pt2(rightUs1[m.queryIdx], kp1.pt.y);
                cosTheta1 = computeCosTheta(cv::Mat::eye(3, 3, CV_32F), cv::Mat::eye(3, 3, CV_32F), kp1.pt, pt2);
            }
            if (depths2[m.trainIdx] > 0) {
                bStereo2 = true;
                cv::Point2f pt2(rightUs2[m.trainIdx], kp2.pt.y);
                cosTheta2 = computeCosTheta(cv::Mat::eye(3, 3, CV_32F), cv::Mat::eye(3, 3, CV_32F), kp2.pt, pt2);
            }
            float cosThetaStereo = std::min(cosTheta1, cosTheta2);
            MapPoint::SharedPtr pMp;
            if (cosTheta0 < cosThetaStereo && cosTheta0 > 0 && (bStereo1 || bStereo2 || cosTheta0 < 0.9998)) {
                cv::Mat p3dW = triangulate(mpCurrKeyFrame, match.first, kp1, kp2);
                if (!p3dW.empty()) {
                    pMp = MapPoint::create(p3dW);
                }
            } else if (bStereo1 && cosTheta1 < cosTheta2) {
                if (mmUnprocessMps.find(m.queryIdx) != mmUnprocessMps.end()) {
                    pMp = mmUnprocessMps[m.queryIdx];
                    mmUnprocessMps.erase(m.queryIdx);
                }
            } else if (bStereo2 && cosTheta2 < cosTheta1) {
                cv::Mat p3dC = match.first->unProject(m.trainIdx);
                cv::Mat Rwc, twc;
                match.first->getPoseInv(Rwc, twc);
                cv::Mat p3dW = Rwc * p3dC + twc;
                pMp = MapPoint::create(p3dW);
            }

            if (!pMp)
                continue;
            auto ret = pMp->checkMapPoint(mpCurrKeyFrame, match.first, m.queryIdx, m.trainIdx);
            if (ret) {
                /// 匹配不会放弃，只不过这里会选择性的使用不同方式产生的地图点
                mpCurrKeyFrame->setMapPoint(m.queryIdx, pMp);
                match.first->setMapPoint(m.trainIdx, pMp);
                pMp->addAttriInit(mpCurrKeyFrame, m.queryIdx);
                pMp->addObservation(match.first, m.trainIdx);
                pMp->updateDescriptor();
                pMp->updateNormalAndDepth();
                mpMap->insertMapPoint(pMp, mpMap);
                mlpAddedMPs.push_back(pMp);
            }
        }
    }

    /// 处理自己本身可以产生的地图点
    auto vMps = mpCurrKeyFrame->getMapPoints();
    for (const auto &item : mmUnprocessMps) {
        auto &pMp = vMps[item.first];
        if (!pMp || pMp->isBad()) {
            mpCurrKeyFrame->setMapPoint(item.first, item.second);
            item.second->addAttriInit(mpCurrKeyFrame, item.first);
            mpMap->insertMapPoint(item.second, mpMap);
            mlpAddedMPs.push_back(item.second);
        }
    }
}

/**
 * @brief 计算两匹配点之间的cosTheta值，判断theta的大小
 *
 * @param R1w 输入的匹配点1对应的旋转矩阵
 * @param R2w 输入的匹配点2对应的旋转矩阵
 * @param pt1 输入的匹配点1对应的坐标
 * @param pt2 输入的匹配点2对应的坐标
 * @return float 输出的cosTheta值
 */
float LocalMapping::computeCosTheta(cv::Mat R1w, cv::Mat R2w, const cv::Point2f &pt1, const cv::Point2f &pt2) {
    float x1 = (pt1.x - Camera::mfCx) / Camera::mfFx;
    float y1 = (pt1.y - Camera::mfCy) / Camera::mfFy;
    float x2 = (pt2.x - Camera::mfCx) / Camera::mfFx;
    float y2 = (pt2.y - Camera::mfCy) / Camera::mfFy;
    cv::Mat pt1W = R1w.t() * (cv::Mat_<float>(3, 1) << x1, y1, 1);
    cv::Mat pt2W = R2w.t() * (cv::Mat_<float>(3, 1) << x2, y2, 1);
    return pt1W.dot(pt2W) / (cv::norm(pt1W) * cv::norm(pt2W));
}

/**
 * @brief 使用SVD矩阵分解三角化
 * @details
 *      1. 如果SVD失败，输出空矩阵
 * @param pkf1  输入的关键帧1
 * @param pkf2  输入的关键帧2
 * @param kp1   输入的匹配点1
 * @param kp2   输入的匹配点2
 * @return cv::Mat 输出的三角化点
 */
cv::Mat LocalMapping::triangulate(KeyFrame::SharedPtr pkf1, KeyFrame::SharedPtr pkf2, const cv::KeyPoint &kp1,
                                  const cv::KeyPoint &kp2) {
    cv::Mat T1w = pkf1->getPose();
    cv::Mat T2w = pkf2->getPose();
    cv::Mat R1w = T1w.rowRange(0, 3).colRange(0, 3);
    cv::Mat t1w = T1w.rowRange(0, 3).col(3);
    cv::Mat R2w = T2w.rowRange(0, 3).colRange(0, 3);
    cv::Mat t2w = T2w.rowRange(0, 3).col(3);
    cv::Mat A(4, 4, CV_32F);
    A(cv::Range(0, 1), cv::Range(0, 3)) = Camera::mfFx * R1w.row(0) + (Camera::mfCx - kp1.pt.x) * R1w.row(2);
    A(cv::Range(1, 2), cv::Range(0, 3)) = Camera::mfFy * R1w.row(1) + (Camera::mfCy - kp1.pt.y) * R1w.row(2);
    A(cv::Range(2, 3), cv::Range(0, 3)) = Camera::mfFx * R2w.row(0) + (Camera::mfCx - kp2.pt.x) * R2w.row(2);
    A(cv::Range(3, 4), cv::Range(0, 3)) = Camera::mfFy * R2w.row(1) + (Camera::mfCy - kp2.pt.y) * R2w.row(2);
    A.at<float>(0, 3) = Camera::mfFx * t1w.at<float>(0) + (Camera::mfCx - kp1.pt.x) * t1w.at<float>(2);
    A.at<float>(1, 3) = Camera::mfFy * t1w.at<float>(1) + (Camera::mfCy - kp1.pt.y) * t1w.at<float>(2);
    A.at<float>(2, 3) = Camera::mfFx * t2w.at<float>(0) + (Camera::mfCx - kp2.pt.x) * t2w.at<float>(2);
    A.at<float>(3, 3) = Camera::mfFy * t2w.at<float>(1) + (Camera::mfCy - kp2.pt.y) * t2w.at<float>(2);
    cv::Mat u, w, vt, ret(3, 1, CV_32F);
    cv::SVD::compute(A, w, u, vt);
    if (w.at<float>(3) / w.at<float>(2) > 1e-3)
        return cv::Mat();
    cv::Mat tmp = vt.row(3) / vt.at<float>(3, 3);
    if (tmp.at<float>(2) < 0)
        return cv::Mat();
    ret.at<float>(0) = tmp.at<float>(0);
    ret.at<float>(1) = tmp.at<float>(1);
    ret.at<float>(2) = tmp.at<float>(2);
    return ret;
}

/**
 * @brief 对地图点新增之后，进行地图点的融合操作
 * @details
 *      1. 取参与融合的关键帧和地图点（10个一阶+5个二阶）
 *      2. 进行正向投影融合（将参与融合的地图点都投影到当前关键帧中）
 *      3. 进行反向投影融合（将当前关键帧中的地图点投影到参与融合的关键帧中）
 *      4. 更新当前关键帧的连接关系
 * @note
 *      1. 在正向投影融合中，会初步进行地图点的筛选
 *      2. 在投影结束后，会进行融合操作（未产生的匹配进行添加，重复的匹配进行替换）
 */
void LocalMapping::fuseMapPoints() {
    std::unordered_set<KeyFrame::SharedPtr> sTargetKfs;
    std::unordered_set<MapPoint::SharedPtr> sTargetMps;
    std::vector<MapPoint::SharedPtr> vTargetMps;
    sTargetKfs.insert(mpCurrKeyFrame);
    auto connectedKfs = mpCurrKeyFrame->getOrderedConnectedKfs(10);
    for (auto item : connectedKfs) {
        int nNum = 0;
        auto connectedKfs2 = item->getOrderedConnectedKfs(100);
        sTargetKfs.insert(item);
        for (auto pkf : connectedKfs2) {
            if (sTargetKfs.find(pkf) == sTargetKfs.end()) {
                sTargetKfs.insert(pkf);
                ++nNum;
                if (nNum == 5) {
                    break;
                }
            }
        }
    }
    for (auto &pkf : sTargetKfs) {
        auto mps = pkf->getMapPoints();
        for (auto &pMp : mps) {
            if (!pMp || pMp->isBad())
                continue;
            sTargetMps.insert(pMp);
        }
    }
    std::unordered_set<MapPoint::SharedPtr> sNoMps;
    auto mps = mpCurrKeyFrame->getMapPoints();
    for (auto &pMp : mps) {
        if (!pMp || pMp->isBad())
            continue;
        sNoMps.insert(pMp);
    }
    auto sNoMpsEnd = sNoMps.end();
    std::copy_if(sTargetMps.begin(), sTargetMps.end(), std::back_inserter(vTargetMps),
                 [&sNoMps, &sNoMpsEnd](MapPoint::SharedPtr pMp) { return sNoMps.find(pMp) == sNoMpsEnd; });
    ORBMatcher matcher(0.6, true);
    int nFuse = matcher.fuse(mpCurrKeyFrame, vTargetMps, mpMap);
    int nFuseInv = 0;
    for (auto &pkf : sTargetKfs)
        nFuseInv += matcher.fuse(pkf, mpCurrKeyFrame, mpMap);

    KeyFrame::updateConnections(mpCurrKeyFrame);
}

/**
 * @brief 删除冗余关键帧
 * @details
 *      1. 删除范围：与当前关键帧产生连接的一阶关键帧
 *      2. 做一个地图点数据库，来统计这些关键帧中被观测的地图点信息
 *      3. 对关键帧进行遍历，当冗余地图点占有90%时，关键帧被删除
 *          1. 删除与该关键帧产生共视关系（关键帧和关键帧之间）
 *          2. 删除与该关键帧产生观测的关系（关键帧和地图点之间）
 *          3. 处理删除关键帧的父子关系（为子关键帧寻找新的父关键帧）
 *          4. 在地图中，删除待删除的关键帧
 * @note 有些关键帧是不能被删除的
 *      1. 永远不能被删除的关键帧：第0个关键帧
 *      2. 暂时不能被删除的关键帧：正在参与回环闭合的关键帧
 */
void LocalMapping::cullingKeyFrames() {
    auto vpTargetKfs = mpCurrKeyFrame->getConnectedKfs(0);
    vpTargetKfs.push_back(mpCurrKeyFrame);
    MapPointDB mapPointDB;
    createMpsDB(vpTargetKfs, mapPointDB);
    for (auto iter = mspToBeErased.begin(); iter != mspToBeErased.end();) {
        auto pkf = *iter;
        if (pkf->isNotErased() || pkf == mpCurrKeyFrame) {
            ++iter;
        } else {
            iter = mspToBeErased.erase(iter);
            deleteKeyFrame(pkf);
        }
    }

    for (auto &pkf : vpTargetKfs) {
        if (judgeKeyFrame(pkf, mapPointDB)) {
            if (pkf->getID() == 0)
                continue;
            if (pkf->isNotErased()) {
                mspToBeErased.insert(pkf);
                continue;
            }
            deleteKeyFrame(pkf);
        }
    }
}

/**
 * @brief 删除给定关键帧pkf
 * @details
 *      1. 删除与该关键帧产生共视关系（关键帧和关键帧之间）
 *          1. 方式一：使用关键帧的共视关系删除
 *              1. 删除不完全，因为在pkf之后产生共视的关键帧不会同步给pkf
 *          2. 方式二：使用关键帧的地图点关系删除
 *      2. 删除与该关键帧产生观测的关系（关键帧和地图点之间）
 *      3. 处理删除关键帧的父子关系（为子关键帧寻找新的父关键帧）
 *      4. 值得注意的是，这里是整个系统中唯一删除关键帧的地方
 *          1. 在函数中，待删除的关键帧的父关键帧和子关键帧都是合理的，非bad的
 *          2. 因此不存在关键帧的父关键帧和子关键帧非法的情况
 * @param pkf 输入的待删除的关键帧pkf
 */
void LocalMapping::deleteKeyFrame(KeyFramePtr &pkf) {
    auto mapPoints = pkf->getMapPoints();
    std::size_t nNum = mapPoints.size();
    for (std::size_t idx = 0; idx < nNum; ++idx) {
        MapPoint::SharedPtr pMp = mapPoints[idx];
        if (!pMp || pMp->isBad())
            continue;
        auto obss = pMp->getObservation();
        for (auto obs : obss) {
            KeyFrame::SharedPtr pkfi = obs.first.lock();
            if (pkfi == pkf) {
                pMp->eraseObservetion(pkf);
                continue;
            }
            if (pkfi && !pkfi->isBad())
                pkfi->eraseConnection(pkf);
        }
    }

    auto spChildren = pkf->getChildren();
    auto pParent = pkf->getParent().lock();
    pParent->eraseChild(pkf);
    for (auto iter = spChildren.begin(); iter != spChildren.end();) {
        auto child = iter->lock();
        pkf->eraseChild(child);
        if (!child || child->isBad()) {
            iter = spChildren.erase(iter);
            continue;
        }
        child->setParent(nullptr);
        ++iter;
    }
    assert(pParent && !pParent->isBad() && "这里的父关键帧应该都是合法的");
    std::vector<KeyFrame::SharedPtr> vpCandidates{pParent};
    if (!spChildren.empty())
        findParent(spChildren, vpCandidates);
    pkf->setBad();
    mpMap->eraseKeyFrame(pkf);
}

/**
 * @brief 使用最小生成树的方法，为失去父关键帧的关键帧寻找父关键帧
 * @details
 *      1. 找到子关键帧中，与候选父关键帧中最大连接权重的匹配（使用一阶相连关键帧寻找）
 *      2. 设置匹配成功的关键帧作为父子关键帧，并将子关键帧放入候选父关键帧中
 *      3. 循环上述过程，直到没有子关键帧为止
 *      4. 会产生一种现象，就是子关键帧没有连接权重（使用待删除关键帧的父关键帧作为他们的父关键帧）
 * @param spChildren    输入输出的子关键帧
 * @param vpCandidates  输入输出的候选父关键帧
 */
void LocalMapping::findParent(std::set<KeyFrameWeak, KeyFrame::WeakCompareFunc> &spChildren,
                              std::vector<KeyFramePtr> &vpCandidates) {
    while (!spChildren.empty()) {
        WeightDB weightDB;
        for (const auto &childWeak : spChildren) {
            KeyFrame::SharedPtr child = childWeak.lock();
            if (!child || child->isBad()) {
                spChildren.erase(childWeak);
                continue;
            }
            auto mConnected = child->getAllConnected();
            std::size_t maxWeight = 0;
            KeyFramePtr pBestCandidate = nullptr;
            for (auto &pCandidate : vpCandidates) {
                auto iter = mConnected.find(pCandidate);
                if (iter == mConnected.end())
                    continue;
                else {
                    if (iter->second > maxWeight) {
                        maxWeight = iter->second;
                        pBestCandidate = pCandidate;
                    }
                }
            }
            if (maxWeight > 0)
                weightDB.insert(std::make_pair(maxWeight, std::make_pair(child, pBestCandidate)));
        }
        if (weightDB.empty()) {
            for (const auto &childWeak : spChildren) {
                auto child = childWeak.lock();
                child->setParent(vpCandidates[0]);
                vpCandidates[0]->addChild(child);
            }
            break;
        } else {
            auto child = weightDB.begin()->second.first;
            auto pCandidate = weightDB.begin()->second.second;
            spChildren.erase(child);
            vpCandidates.push_back(child);
            child->setParent(pCandidate);
            pCandidate->addChild(child);
        }
    }
}

/**
 * @brief 判断某个关键帧是否冗余，90%的观测为冗余则关键帧冗余
 *
 * @param pkf           输入的待判断的关键帧
 * @param mapPointDB    输入的地图点数据库
 * @return true         冗余
 * @return false        不冗余
 */
bool LocalMapping::judgeKeyFrame(KeyFramePtr &pkf, const MapPointDB &mapPointDB) {
    if (pkf->getID() == 0)
        return false;

    auto vpMapPoints = pkf->getMapPoints();
    auto &vKeyPoints = pkf->getLeftKeyPoints();
    int nMpNum = 0, nBad = 0;
    for (std::size_t idx = 0; idx < vpMapPoints.size(); ++idx) {
        auto &pMp = vpMapPoints[idx];
        if (!pMp || pMp->isBad())
            continue;
        ++nMpNum;
        auto &kp = vKeyPoints[idx];
        auto iter = mapPointDB.find(pMp);
        if (iter != mapPointDB.end()) {
            if (iter->second[kp.octave] > 3)
                ++nBad;
        }
    }
    float ratio = float(nBad) / float(nMpNum);
    return ratio > 0.9;
}

/**
 * @brief 创建地图点数据库
 *
 * @param vpTargetKfs   输入的目标关键帧
 * @param mapPointDB    输入输出的地图点数据库<地图点, 观测2维关键点的数目组成的vector(金字塔层级为索引)>
 */
void LocalMapping::createMpsDB(std::vector<KeyFramePtr> &vpTargetKfs, MapPointDB &mapPointDB) {
    for (auto &pkf : vpTargetKfs) {
        if (!pkf || pkf->isBad())
            continue;
        auto mapPoints = pkf->getMapPoints();
        for (auto &pMp : mapPoints) {
            std::vector<std::size_t> pyramidVec;
            pyramidVec.resize(ORBExtractor::mnLevels, 0);
            if (!pMp || pMp->isBad())
                continue;
            auto obs = pMp->getObservation();
            for (auto &item : obs) {
                auto pkfObs = item.first.lock();
                if (!pkfObs || pkfObs->isBad()) {
                    pMp->eraseObservetion(pkfObs);
                    continue;
                }
                cv::KeyPoint kp = pkfObs->getLeftKeyPoints()[item.second];
                ++pyramidVec[kp.octave];
            }
            mapPointDB.insert(std::make_pair(pMp, pyramidVec));
        }
    }
    for (auto &item : mapPointDB) {
        std::vector<std::size_t> pyramidVec;
        for (std::size_t idx = 0; idx < ORBExtractor::mnLevels; ++idx) {
            std::size_t maxId = std::min(idx + 1, (std::size_t)ORBExtractor::mnLevels - 1);
            std::size_t nNum = 0;
            for (std::size_t jdx = idx; jdx < maxId + 1; ++jdx)
                nNum += item.second[jdx];
            pyramidVec.push_back(nNum);
        }
        item.second = pyramidVec;
    }
}

/**
 * @brief 删除冗余地图点
 * @details
 *      1. 从地图中删除（地图点不合格）
 *          (1) 地图点在跟踪线程，对位姿优化的贡献小（内点/观测 < 0.25）
 *          (2) 地图点被创建之后，超过(包括)两个关键帧，但是该地图点还是只能被创建它的关键帧观测
 *      2. 从最近添加地图点容器中删除（地图点合格，以后只会更改不会删除）
 *          (1) 地图点建立以来，已经超过3个关键帧，但是没有从地图中删除
 */
void LocalMapping::cullingMapPoints() {
    auto currID = mpCurrKeyFrame->getID();
    for (auto it = mlpAddedMPs.begin(); it != mlpAddedMPs.end();) {
        MapPoint::SharedPtr pMp = *it;
        if (!pMp || pMp->isBad()) {
            it = mlpAddedMPs.erase(it);
            mpMap->eraseMapPoint(pMp);
            continue;
        }
        float ratio = pMp->scoreInTrack();
        if (ratio < 0.25) {
            it = mlpAddedMPs.erase(it);
            pMp->setBad();
            mpMap->eraseMapPoint(pMp);
            continue;
        }
        KeyFrame::SharedPtr refKF = pMp->getRefKF();
        if (!refKF || refKF->isBad()) {
            pMp->updateNormalAndDepth();
            refKF = pMp->getRefKF();
        }
        if (!refKF || refKF->isBad() || (currID - refKF->getID() >= 2 && pMp->getObsNum() == 1)) {
            it = mlpAddedMPs.erase(it);
            pMp->setBad();
            mpMap->eraseMapPoint(pMp);
            continue;
        }
        if (refKF && !refKF->isBad() && currID - refKF->getID() > 3) {
            it = mlpAddedMPs.erase(it);
            continue;
        }
        ++it;
    }
}

/**
 * @brief 插入关键帧
 *
 * @param pkf 输入的待插入局部建图线程的关键帧
 */
void LocalMapping::insertKeyFrame(KeyFrame::SharedPtr pkf) {
    std::unique_lock<std::mutex> lock(mMutexQueue);
    mqpKeyFrames.push(pkf);
    mbAbortBA = true;
}

void LocalMapping::addKF2DB(KeyFrameDBPtr pkfDB) {
    if (mpCurrKeyFrame && !mpCurrKeyFrame->isBad())
        pkfDB->addKeyFrame(mpCurrKeyFrame);
}

} // namespace ORB_SLAM2_ROS2