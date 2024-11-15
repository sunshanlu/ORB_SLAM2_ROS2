#include <rclcpp/logger.hpp>

#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/LocalMapping.h"
#include "ORB_SLAM2/LoopClosing.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/Optimizer.h"
#include "ORB_SLAM2/Sim3Solver.h"
#include "ORB_SLAM2/Tracking.h"

using namespace std::chrono_literals;

namespace ORB_SLAM2_ROS2 {

/// 构造函数
LoopClosing::LoopClosing(KeyFrameDBPtr pKeyFrameDB, MapPtr pMap, LocalMappingPtr pLocalMapping, TrackingPtr pTracking)
    : mpKeyFrameDB(pKeyFrameDB)
    , mpMap(pMap)
    , mpLocalMapper(pLocalMapping)
    , mpTracker(pTracking)
    , mbStop(false)
    , mbResquestStop(false)
    , mbStopGBA(false)
    , mnLastLoopId(0) {}

/// 回环闭合线程入口函数
void LoopClosing::run() {
    while (!isRequestStop() && rclcpp::ok()) {
        runOnce();
        if (mpCurrKeyFrame)
            mpCurrKeyFrame->setNotErased(false);
        std::this_thread::sleep_for(3ms);
    }
    mbStopGBA = true;
    if (mpGlobalBAThread) {
        mpGlobalBAThread->join();
        delete mpGlobalBAThread;
        mpGlobalBAThread = nullptr;
    }
    release();
    stop();
}

/// 循环运行一次
void LoopClosing::runOnce() {
    if (!processNewKeyFrame())
        return;

    bool isDetected = detectLoop();
    mpKeyFrameDB->addKeyFrame(mpCurrKeyFrame);

    if (!isDetected)
        return;

    for (auto &pCandidate : mvEnoughKfs) {
        pCandidate->setNotErased(true);
    }

    /// 计算SIM3
    Sim3Ret g2oScm, g2oScw;
    KeyFrame::SharedPtr pMatchKF;
    bool isComputed = computeSim3(g2oScm, g2oScw, pMatchKF);
    for (auto &pCandidate : mvEnoughKfs) {
        if (pCandidate == pMatchKF)
            continue;
        pCandidate->setNotErased(false);
    }
    if (!isComputed)
        return;

    RCLCPP_INFO(rclcpp::get_logger("ORB_SLAM2"), "检测到回环闭合");
    correctLoop(g2oScw, pMatchKF);

    RCLCPP_INFO(rclcpp::get_logger("ORB_SLAM2"), "开始执行全局BA");
    mbStopGBA = false;
    mpGlobalBAThread = new std::thread(&LoopClosing::runGlobalBA, this);

    release();
}

void LoopClosing::runGlobalBA() {
    auto vpOldKfs = mpMap->getAllKeyFrames();
    auto vpOldMps = mpMap->getAllMapPoints();
    Optimizer::globalOptimization(vpOldKfs, vpOldMps, 10, &mbStopGBA, false);

    if (mbStopGBA)
        return;

    mpLocalMapper->requestStop();
    while (!mpLocalMapper->isStop())
        std::this_thread::sleep_for(1ms);

    auto vpNewKfs = mpMap->getAllKeyFrames();
    auto vpNewMps = mpMap->getAllMapPoints();

    /// 处理全局BA后的关键帧
    std::list<KeyFrame::SharedPtr> qToProcess(vpOldKfs.begin(), vpOldKfs.end());
    while (!qToProcess.empty()) {
        auto &pParent = qToProcess.front();
        if (!pParent || pParent->isBad()) {
            qToProcess.pop_front();
            continue;
        }
        cv::Mat Twp = pParent->getPoseInv();
        if (pParent->mTcwGBA.empty())
            throw std::runtime_error("KeyFrame::mTcwGBA is empty");
        auto vpChildren = pParent->getChildren();
        for (auto &childWeak : vpChildren) {
            auto child = childWeak.lock();
            if (!child || child->isBad())
                continue;
            if (!child->mTcwGBA.empty())
                continue;
            cv::Mat Tcp = child->getPose() * Twp;
            child->mTcwGBA = Tcp * pParent->mTcwGBA;
            qToProcess.push_back(child);
        }
        pParent->mTcwBefGBA = pParent->getPose().clone();
        pParent->setPose(pParent->mTcwGBA);
        qToProcess.pop_front();
    }

    /// 处理全局BA后的地图点
    for (auto &pMp : vpNewMps) {
        if (!pMp || pMp->isBad())
            continue;
        if (!pMp->mPGBA.empty())
            pMp->setPos(pMp->mPGBA);
        else {
            auto refKF = pMp->getRefKF();
            if (!refKF || refKF->isBad()) {
                pMp->updateNormalAndDepth();
                refKF = pMp->getRefKF();
                if (!refKF || refKF->isBad())
                    continue;
            }
            if (refKF->mTcwBefGBA.empty()) {
                throw std::runtime_error("mTcwBefGBA 为空");
            }
            cv::Mat RcwBef, tcwBef, Rwc, twc;
            refKF->getPoseInv(Rwc, twc);
            refKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3).copyTo(RcwBef);
            refKF->mTcwBefGBA.rowRange(0, 3).col(3).copyTo(tcwBef);
            cv::Mat P3dc = RcwBef * pMp->getPos() + tcwBef;
            pMp->setPos(Rwc * P3dc + twc);
        }
    }
    RCLCPP_INFO(rclcpp::get_logger("ORB_SLAM2"), "地图已经完成了更新");
    mpLocalMapper->start();
}

/**
 * @brief 当完成一次回环后，需要释放当前回环线程的资源
 *
 */
void LoopClosing::release() {
    std::queue<KeyFramePtr> empty;
    {
        std::unique_lock<std::mutex> lock(mMutexQueue);
        std::swap(mqKeyFrames, empty);
    }
    mvConsistGroups.clear();
    mvEnoughKfs.clear();
    mvLoopGroupMps.clear();
    mvMatchedMps.clear();
}

/**
 * @brief 处理新关键帧，保证新关键帧不能被删除
 *
 */
bool LoopClosing::processNewKeyFrame() {
    {
        std::unique_lock<std::mutex> lock(mMutexQueue);
        if (mqKeyFrames.empty())
            return false;

        mpCurrKeyFrame = mqKeyFrames.front();
        mqKeyFrames.pop();
    }
    mpCurrKeyFrame->setNotErased(true);
    return true;
}

/**
 * @brief 判断是否有满足回环闭合关键帧存在
 * @details
 *      1. 统计候选组与连续组之间的连续性关系
 *          1) 如果发生连续，更新当前的连续组链的内容和长度
 *          2) 注意，发生过连续性条件的连续组不会重复进行检测
 *      2. 检测连续性条件，即检测当前连续组链中，是否有满足连续性条件的连续组
 *          1) 如果有满足连续性条件的连续组，将连续组对应的候选关键帧加入到mvEnoughKfs中，返回true
 *          2) 如果没有满足连续性条件的连续组，返回false
 * @return true     有满足连续性条件的回环闭合关键帧
 * @return false    没有满足连续性条件的回环闭合关键帧
 */
bool LoopClosing::detectLoop() {
    mvEnoughKfs.clear();
    if (mpCurrKeyFrame->getID() < mnLastLoopId + 10)
        return false;
    std::vector<KeyFrame::SharedPtr> vpLoopCandidates;
    mpKeyFrameDB->findLoopCloseKfs(mpCurrKeyFrame, vpLoopCandidates);
    if (vpLoopCandidates.empty())
        return false;

    decltype(mvConsistGroups) vCurrConsistGroups;

    bool groupEmpty = mvConsistGroups.empty();
    std::vector<bool> vbConsist(mvConsistGroups.size(), false);
    for (auto &pCandidate : vpLoopCandidates) {
        auto vCandidateConnecteds = pCandidate->getConnectedKfs(0);
        std::set<KeyFrame::SharedPtr> sCandidateGroup(vCandidateConnecteds.begin(), vCandidateConnecteds.end());
        sCandidateGroup.insert(pCandidate);

        if (groupEmpty) {
            vCurrConsistGroups.push_back(std::make_pair(sCandidateGroup, 0));
            continue;
        }

        bool isConsist = false;
        for (std::size_t idx = 0; idx < mvConsistGroups.size(); ++idx) {
            if (vbConsist[idx])
                continue;
            auto &sConsistGroup = mvConsistGroups[idx].first;
            int previousLen = mvConsistGroups[idx].second;
            if (sConsistGroup.size() > sCandidateGroup.size())
                isConsist = isConsistBetween(sConsistGroup, sCandidateGroup);
            else
                isConsist = isConsistBetween(sCandidateGroup, sConsistGroup);
            if (isConsist) {
                vCurrConsistGroups.push_back(std::make_pair(sCandidateGroup, previousLen + 1));
                vbConsist[idx] = true;
                break;
            }
        }
        if (!isConsist) {
            vCurrConsistGroups.push_back(std::make_pair(sCandidateGroup, 0));
        }
    }

    for (std::size_t idx = 0; idx < vCurrConsistGroups.size(); ++idx) {
        auto &currConsistGroup = vCurrConsistGroups[idx];
        auto &candidateKf = vpLoopCandidates[idx];
        if (currConsistGroup.second >= 3) {
            mvEnoughKfs.push_back(candidateKf);
        }
    }
    if (mvEnoughKfs.empty())
        std::swap(mvConsistGroups, vCurrConsistGroups);
    else
        mvConsistGroups.clear();
    return !mvEnoughKfs.empty();
}

/**
 * @brief 计算SIM3矩阵，使用RANSAC+SIM3求解器的方法
 * @details
 *      1. 使用词袋匹配，将匹配数目少于20的作为discard丢弃（粗筛）
 *      2. 使用雨露均沾打发，进行RANSAC和求解器的闭式求解
 *      3. 如果RANSAC求解成功
 *          1. 如果bNoMore为false，是比较好的解，使用RANSAC的先验内点信息（只作用一回迭代）
 *          2. 如果bNoMore为true，是比较差的解，不使用RANSAC的先验内点信息（只作用一回迭代）
 *          3. 如果bNoMore为false，是比较好的解，使用SIM3重投影，阈值为7.5，增加匹配信息
 *          4. 如果bNoMore为true，是比较差的解，使用SIM3重投影匹配，阈值为15，增加匹配信息
 *      4.
 * @param g2oScm    输出的经过g2o优化的SIM3变换矩阵
 * @param g2oScw    输入的初始的SIM3变换矩阵
 * @return true     计算SIM3成功
 * @return false    计算SIM3失败
 */
bool LoopClosing::computeSim3(Sim3Ret &g2oScm, Sim3Ret &g2oScw, KeyFramePtr &pLoopKf) {
    int n = mvEnoughKfs.size();
    ORBMatcher matcher(0.75, true);
    std::vector<bool> vbDiscard(n, false);
    std::vector<Sim3Solver::SharedPtr> vpSolvers(n, nullptr);
    std::vector<std::vector<cv::DMatch>> vvMatches(n);
    int nCandidates = n;
    for (std::size_t idx = 0; idx < n; ++idx) {
        auto pKfCandidate = mvEnoughKfs[idx];
        if (!pKfCandidate || pKfCandidate->isBad()) {
            vbDiscard[idx] = true;
            --nCandidates;
            continue;
        }
        std::vector<cv::DMatch> matches;
        int nMatches = matcher.searchByBow(mpCurrKeyFrame, pKfCandidate, matches, false, true);
        if (nMatches < 20) {
            vbDiscard[idx] = true;
            --nCandidates;
            continue;
        }
        std::vector<bool> vbChoose(matches.size(), true);
        vpSolvers[idx] = Sim3Solver::create(pKfCandidate, mpCurrKeyFrame, matches, vbChoose);
        int nChoose = 0;
        for (std::size_t jdx = 0; jdx < vbChoose.size(); ++jdx) {
            if (vbChoose[jdx]) {
                vvMatches[idx].push_back(matches[jdx]);
                ++nChoose;
            }
        }
        if (nChoose < 20) {
            vbDiscard[idx] = true;
            --nCandidates;
            continue;
        }
    }
    bool bComplete = false;
    KeyFrame::SharedPtr pkfCandidate;
    std::vector<cv::DMatch> vMatches;
    while (!bComplete && nCandidates > 0) {
        for (std::size_t idx = 0; idx < n; ++idx) {
            pkfCandidate = mvEnoughKfs[idx];
            if (vbDiscard[idx])
                continue;
            Sim3Ret Scm;
            bool bNoMore = false;
            auto pSolver = vpSolvers[idx];
            std::vector<std::size_t> vIndices;
            auto ret = pSolver->iterate(5, g2oScm, bNoMore, vIndices);
            if (bNoMore) {
                vbDiscard[idx] = true;
                --nCandidates;
            }
            if (ret) {
                vMatches.clear();
                int nMatches = 0;
                nMatches = matcher.searchBySim3(mpCurrKeyFrame, pkfCandidate, vMatches, g2oScm, 7.5);
                if (nMatches < 50)
                    continue;
                int nInlier = Optimizer::OptimizeSim3(mpCurrKeyFrame, pkfCandidate, vMatches, g2oScm);
                if (nInlier > 50) {
                    bComplete = true;
                    break;
                }
            }
        }
    }
    if (!bComplete)
        return false;
    auto vLoopGroup = pkfCandidate->getConnectedKfs(0);
    std::set<KeyFrame::SharedPtr> sLoopGroup(vLoopGroup.begin(), vLoopGroup.end());
    sLoopGroup.erase(nullptr);
    sLoopGroup.insert(pkfCandidate);
    mvLoopGroupMps.clear();

    std::set<MapPointPtr> sLoopGroupMps;
    for (auto &pKf : sLoopGroup) {
        auto vMapPointsi = pKf->getMapPoints();
        for (auto &pMp : vMapPointsi) {
            if (pMp && !pMp->isBad() && pMp->isInMap())
                sLoopGroupMps.insert(pMp);
        }
    }
    cv::Mat Rmw, tmw;
    pkfCandidate->getPose(Rmw, tmw);
    Sim3Ret g2oSmw(Rmw, tmw, 1.0);
    g2oScw = g2oScm * g2oSmw;
    mvMatchedMps.clear();
    mvMatchedMps.resize(mpCurrKeyFrame->getLeftKeyPoints().size(), nullptr);
    for (const auto &m : vMatches) {
        auto pMpM = pkfCandidate->getMapPoint(m.trainIdx);
        mvMatchedMps[m.queryIdx] = pMpM;
    }
    for (const auto &pMpM : sLoopGroupMps)
        mvLoopGroupMps.push_back(pMpM);
    int nMatches = matcher.searchBySim3(mpCurrKeyFrame, mvLoopGroupMps, mvMatchedMps, g2oScw, 10);
    if (nMatches < 40)
        return false;
    pLoopKf = pkfCandidate;
    return true;
}

/**
 * @brief 矫正回环
 * @details
 *      1. 闭环关键帧的位姿矫正，对象是当前关键帧的相连关键帧
 *      2. 闭环地图点的位置矫正，对象是当前关键帧的地图点
 *      3. 地图点的投影融合，msLoopGroupMps投影融合到当前关键帧组中来
 *      4. 更新当前关键的两级共视关键帧的连接关系，统计新添加的连接关系
 *      5. 统计新生成的关键帧之间的连接关系
 *          1. 获取当前关键帧组的一阶相连关键帧，组成更大的当前关键帧组
 *          2. 统计关键帧组之间的连接关系
 *      5. 进行本质图优化
 * @param g2oScw    输入的当前关键帧的SIM3变换矩阵
 * @return true     回环矫正成功
 * @return false    回环矫正失败
 */
void LoopClosing::correctLoop(const Sim3Ret &g2oScw, const KeyFramePtr &pLoopKf) {
    /// 请求局部建图线程停止
    mpLocalMapper->requestStop();
    while (!mpLocalMapper->isStop())
        std::this_thread::sleep_for(1ms);

    /// 请求全局BA停止
    mbStopGBA = true;
    if (mpGlobalBAThread) {
        mpGlobalBAThread->detach();
        delete mpGlobalBAThread;
        mpGlobalBAThread = nullptr;
    }

    /// 同步局部建图线程
    KeyFrame::updateConnections(mpCurrKeyFrame);

    /// 矫正当前关键帧组位姿
    KeyFrameAndSim3 correctedSim3, unCorrectedSim3;
    auto vConnectedKfs = mpCurrKeyFrame->getConnectedKfs(0);
    vConnectedKfs.push_back(mpCurrKeyFrame);
    cv::Mat Rwc, twc;
    mpCurrKeyFrame->getPoseInv(Rwc, twc);
    Sim3Ret Swc(Rwc, twc, 1.f);
    {
        std::unique_lock<std::mutex> lock(mpMap->getGlobalMutex());
        for (auto &pKf : vConnectedKfs) {
            cv::Mat Riw, tiw;
            pKf->getPose(Riw, tiw);
            Sim3Ret Siw(Riw, tiw, 1.f);
            Sim3Ret Sic = Siw * Swc;
            Sim3Ret g2oSiw = Sic * g2oScw;
            correctedSim3.insert({pKf, g2oSiw});
            unCorrectedSim3.insert({pKf, Siw});
            pKf->setPose(g2oSiw.mRqp, g2oSiw.mtqp / g2oSiw.mfS);
        }

        /// 矫正当前关键帧组地图点
        for (auto &pKf : vConnectedKfs) {
            const auto &Siw = unCorrectedSim3[pKf];
            const auto &g2oSiw = correctedSim3[pKf];
            auto vMapPoints = pKf->getMapPoints();
            for (auto &pMp : vMapPoints) {
                if (!pMp || pMp->isBad() || !pMp->isInMap())
                    continue;
                if (pMp->getLoopKF() == mpCurrKeyFrame)
                    continue;
                auto p3dW = pMp->getPos();
                auto p3dC = Siw * p3dW;
                p3dW = g2oSiw.inv() * p3dC;
                pMp->setPos(p3dW);
                pMp->setLoopKF(mpCurrKeyFrame);
                pMp->setLoopRefKF(pKf);
                pMp->updateNormalAndDepth();
            }
            KeyFrame::updateConnections(pKf);
        }

        /// 回环闭合关键帧组投影到当前关键帧组中去（地图点的融合）
        vConnectedKfs.pop_back();
        std::size_t nMatchesCurr = mvMatchedMps.size();
        for (std::size_t idx = 0; idx < nMatchesCurr; ++idx) {
            auto pMpC = mpCurrKeyFrame->getMapPoint(idx);
            auto pMpM = mvMatchedMps[idx];
            if (!pMpM || pMpM->isBad() || !pMpM->isInMap())
                continue;
            if (pMpC && !pMpC->isBad() && pMpC->isInMap())
                MapPoint::replace(pMpC, mvMatchedMps[idx], mpMap);
            else {
                pMpM->addObservation(mpCurrKeyFrame, idx);
                mpCurrKeyFrame->setMapPoint(idx, pMpM);
            }
        }
        mpMap->setUpdate(true);
    }

    ORBMatcher matcher(0.8, true);
    for (auto &pKf : vConnectedKfs)
        matcher.fuse(pKf, mvLoopGroupMps, mpMap, true, 4.0f);

    /// 统计新生成的关键帧之间的连接关系
    vConnectedKfs.push_back(mpCurrKeyFrame);
    std::map<KeyFramePtr, std::set<KeyFramePtr>> mLoopConnections;
    for (auto &pKf : vConnectedKfs) {
        auto vPreviousConnections = pKf->getConnectedKfs(0);
        KeyFrame::updateConnections(pKf);
        auto vCurrentConnections = pKf->getConnectedKfs(0);
        mLoopConnections.insert({pKf, std::set<KeyFramePtr>(vCurrentConnections.begin(), vCurrentConnections.end())});
        for (auto &pKfPrevious : vPreviousConnections)
            mLoopConnections[pKf].erase(pKfPrevious);
        for (auto &pkfi : vConnectedKfs)
            mLoopConnections[pKf].erase(pkfi);
    }
    mpCurrKeyFrame->addLoopEdge(pLoopKf);

    /// 优化本质图
    Optimizer::optimizeEssentialGraph(mLoopConnections, mpMap, pLoopKf, mpCurrKeyFrame, 100, correctedSim3,
                                      unCorrectedSim3);
    RCLCPP_INFO(rclcpp::get_logger("ORB_SLAM2"), "本质图优化完成");
    mnLastLoopId = mpCurrKeyFrame->getID();
    pLoopKf->setNotErased(false);
    mpLocalMapper->start();
}

/**
 * @brief 向回环闭合关键帧中队列中插入关键帧（局部建图线程调用）
 *
 * @param pkf 要插入的回环闭合关键帧
 */
void LoopClosing::insertKeyFrame(KeyFramePtr pkf) {
    std::unique_lock<std::mutex> lock(mMutexQueue);
    mqKeyFrames.push(pkf);
}

/**
 * @brief 判断两组关键帧是否满足连续性条件
 * 只要两组关键帧中，有相同元素，那么救认为两组关键帧连续
 * @param groupDB       数据库
 * @param groupFind     查询源
 * @return true     满足连续性条件
 * @return false    不满足连续性条件
 */
bool LoopClosing::isConsistBetween(const std::set<KeyFramePtr> &groupDB, const std::set<KeyFramePtr> &groupFind) {
    auto IterEnd = groupDB.end();
    for (auto &pkf : groupFind) {
        if (groupDB.find(pkf) != IterEnd)
            return true;
    }
    return false;
}

} // namespace ORB_SLAM2_ROS2