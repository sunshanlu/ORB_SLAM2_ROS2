#include "ORB_SLAM2/Tracking.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Optimizer.h"
#include "ORB_SLAM2/PnPSolver.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 处理普通帧
 *
 * @param leftImg   左图像
 * @param rightImg  右图像
 * @return cv::Mat 输出的位姿矩阵
 */
cv::Mat Tracking::grabFrame(cv::Mat leftImg, cv::Mat rightImg) {
    switch (mStatus) {
    case TrackingState::NOT_IMAGE_YET:
        mpCurrFrame = Frame::create(leftImg, rightImg, mnInitFeatures, msBriefTemFp, mnMaxThresh, mnMinThresh);
        break;
    default:
        mpCurrFrame = Frame::create(leftImg, rightImg, mnFeatures, msBriefTemFp, mnMaxThresh, mnMinThresh);
    }
    if (mStatus == TrackingState::NOT_IMAGE_YET) {
        mStatus = TrackingState::NOT_INITING;
        auto pose = cv::Mat::eye(4, 4, CV_32F);
        mpCurrFrame->setPose(pose);
    }

    if (mStatus == TrackingState::NOT_INITING && mpCurrFrame->getN() >= 500) {
        initForStereo();
        mpLastFrame = mpCurrFrame;
        return mpCurrFrame->getPose();
    } else if (mStatus == TrackingState::OK) {
        bool bOK = false;
        mpCurrFrame->setPose(mpLastFrame->getPose());
        if (mVelocity.empty()) {
            bOK = trackReference();
        } else {
            bOK = trackMotionModel();
            if (!bOK)
                bOK = trackReference();
        }
        if (!bOK)
            bOK = trackReLocalize();
        if (bOK) {
            bOK = trackLocalMap();
            if (bOK)
                mStatus = TrackingState::OK;
            else {
                mStatus = TrackingState::LOST;
                return cv::Mat();
            }
        } else {
            mStatus = TrackingState::LOST;
            return cv::Mat();
        }

    } else if (mStatus == TrackingState::LOST) {
        bool bOK = trackReLocalize();
        if (bOK) {
            bOK = trackLocalMap();
            if (bOK)
                mStatus = TrackingState::OK;
        } else {
            mStatus = TrackingState::LOST;
            return cv::Mat();
        }
    }
    updateVelocity();
    updateTlr();
    mpLastFrame = mpCurrFrame;
    return mpCurrFrame->getPose();
}

/// 将当前帧升级为关键帧
KeyFrame::SharedPtr Tracking::updateCurrFrame() { return KeyFrame::create(*mpCurrFrame); }

/// 更新速度Tcl
void Tracking::updateVelocity() {
    cv::Mat Tcw = mpCurrFrame->getPose();
    cv::Mat Twl = mpLastFrame->getPoseInv();
    mVelocity = Tcw * Twl;
}

/// 更新mTlr（上一帧到参考关键帧的位姿）
void Tracking::updateTlr() {
    KeyFrame::SharedPtr refKF = mpCurrFrame->getRefKF();
    if (!refKF || refKF->isBad()) {
        /// 如果参考关键帧被删除了，那么下一次跟踪只能使用参考关键帧跟踪
        mVelocity = cv::Mat();
        return;
    }
    mTlr = mpCurrFrame->getPose() * refKF->getPoseInv();
}

/**
 * @brief 插入关键帧到局部地图中去
 * 插入到局部地图中的关键帧的mbIsLocalKf会被标注为true
 * 同时匹配相同关键的同时插入（判断是否可用）
 */
void Tracking::insertLocalKFrame(KeyFrame::SharedPtr pKf) {
    if (pKf && !pKf->isBad()) {
        auto pParent = pKf->getParent();
        auto vpChildren = pKf->getChildren();
        if (pParent && !pParent->isBad()) {
            if (!pParent->isLocalKf()) {
                pParent->setLocalKf(true);
                mvpLocalKfs.push_back(pParent);
            }
        }
        for (auto &child : vpChildren) {
            if (child && !child->isBad()) {
                if (!child->isLocalKf()) {
                    child->setLocalKf(true);
                    mvpLocalKfs.push_back(child);
                }
            }
        }
        if (pKf->isLocalKf()) {
            return;
        }
        pKf->setLocalKf(true);
        mvpLocalKfs.push_back(pKf);
    }
}

/**
 * @brief 插入地图点到局部地图中去
 * 插入到局部地图中的地图点的mbIsLocalMp会被标注为true
 * 同时避免重复相同地图点的插入（判断是否可用）
 */
void Tracking::insertLocalMPoint(MapPoint::SharedPtr pMp) {
    if (pMp && !pMp->isBad()) {
        if (!pMp->isLocalMp()) {
            pMp->setLocalMp(true);
            mvpLocalMps.push_back(pMp);
        }
    }
}

/**
 * @brief 构建局部地图中的关键帧
 * @details
 *      1. 与当前帧一阶相连的关键帧
 *      2. 与当前帧二阶相连的关键帧
 *      3. 上述关键帧的父亲和儿子关键帧
 *      4. 在插入关键帧之前，需要将之前的局部地图关键帧释放，并将mbIsLocalKf置为false
 */
void Tracking::buildLocalKfs() {
    for (auto &kf : mvpLocalKfs)
        kf->setLocalKf(false);
    mvpLocalKfs.clear();
    auto firstKfs = mpCurrFrame->getConnectedKfs(0);
    for (auto &kf : firstKfs) {
        insertLocalKFrame(kf);
        auto secondKfs = kf->getConnectedKfs(0);
        for (auto &kf2 : secondKfs) {
            insertLocalKFrame(kf2);
        }
    }
}

/**
 * @brief 构建局部地图中的地图点
 * 注意，在使用这个api之前，保证局部地图中的关键帧已经构建完毕
 */
void Tracking::buildLocalMps() {
    for (auto &pMp : mvpLocalMps)
        pMp->setLocalMp(false);
    mvpLocalMps.clear();
    for (auto &kf : mvpLocalKfs) {
        auto mvpMps = kf->getMapPoints();
        for (auto &pMp : mvpMps)
            insertLocalMPoint(pMp);
    }
}

/**
 * @brief 构建局部地图
 * @details
 *      1. 构建局部地图中的关键帧
 *      2. 构建局部地图中的地图点
 */
void Tracking::buildLocalMap() {
    buildLocalKfs();
    buildLocalMps();
}

/**
 * @brief 双目相机的初始化
 *
 */
void Tracking::initForStereo() {
    std::vector<MapPoint::SharedPtr> mapPoints;
    mpCurrFrame->unProject(mapPoints);
    auto kfInit = KeyFrame::create(*mpCurrFrame);

    mpMap->insertKeyFrame(kfInit, mpMap);
    for (std::size_t idx = 0; idx < mapPoints.size(); ++idx) {
        auto &pMp = mapPoints[idx];
        if (!pMp)
            continue;
        pMp->addAttriInit(kfInit, idx);
        mpMap->insertMapPoint(pMp, mpMap);
    }
    mStatus = TrackingState::OK;
    mpRefKf = kfInit;
}

/**
 * @brief 跟踪参考关键帧
 * @details
 *      1. 基于词袋的匹配要求成功匹配大于等于15
 *      2. 基于基于OptimizePoseOnly的优化，要求内点数目大于等于10
 * @return true     跟踪参考关键帧失败
 * @return false    跟踪参考关键帧成功
 */
bool Tracking::trackReference() {
    ORBMatcher matcher(0.7, true);
    std::vector<cv::DMatch> matches;
    int nMatches = matcher.searchByBow(mpCurrFrame, mpRefKf, matches);
    if (nMatches < 15) {
        return false;
    }
    int nInliers = Optimizer::OptimizePoseOnly(mpCurrFrame);

    return nInliers >= 10;
}

/**
 * @brief 基于恒速模型的跟踪
 * @details
 *      1. 基于重投影匹配，初始的半径为15，如果没有合适的匹配，寻找半径变为30
 *      2. OptimizePoseOnly做了外点的剔除（误差较大的边和投影超出图像边界的地图点）
 * @return true     跟踪成功
 * @return false    跟踪失败
 */
bool Tracking::trackMotionModel() {
    std::vector<cv::DMatch> matches;
    processLastFrame();
    mpCurrFrame->setPose(mVelocity * mpLastFrame->getPose());
    ORBMatcher matcher(0.7, true);
    int nMatches = matcher.searchByProjection(mpCurrFrame, mpLastFrame, matches, 15);
    if (nMatches < 20) {
        nMatches += matcher.searchByProjection(mpCurrFrame, mpLastFrame, matches, 30);
    }
    if (nMatches < 20) {
        return false;
    }
    int inLiers = Optimizer::OptimizePoseOnly(mpCurrFrame);
    return inLiers >= 10;
}

/**
 * @brief 重定位第一步：使用关键帧数据库，找到初步的候选关键帧
 *
 * @param vpCandidateKFs    输出的初步候选关键帧
 * @param candidateNum      输出的候选关键帧数量
 * @return true     找到初步的候选关键帧
 * @return false    没找到初步的候选关键帧
 */
bool Tracking::findInitialKF(std::vector<KeyFrame::SharedPtr> &vpCandidateKFs, int &candidateNum) {
    mpKfDB->findRelocKfs(mpCurrFrame, vpCandidateKFs);
    candidateNum = vpCandidateKFs.size();
    if (candidateNum == 0)
        return false;
    return true;
}

/**
 * @brief RelocBowParam 结构体的构造函数
 *
 * @param nCandidate 输入的候选关键帧数量
 */
RelocBowParam::RelocBowParam(int nCandidate) {
    mvpSolvers.resize(nCandidate, nullptr);
    mvAllMatches.resize(nCandidate);
    mvPnPId2MatchID.resize(nCandidate);
}

/**
 * @brief 使用词袋匹配进一步筛选候选关键帧
 *
 * @param relocBowParam     输出的基于词袋匹配的信息
 * @param vbDiscard         输出的不合格关键帧的标识
 * @param candidateNum      输入的候选关键帧数量
 * @param vpCandidateKFs    输入的候选关键帧
 * @return int 输出的合格关键帧数量
 */
int Tracking::filterKFByBow(RelocBowParam &relocBowParam, std::vector<bool> &vbDiscard, const int &candidateNum,
                            std::vector<KeyFrame::SharedPtr> &vpCandidateKFs) {
    int nCandidates = 0;
    for (std::size_t idx = 0; idx < candidateNum; ++idx) {
        KeyFrame::SharedPtr pkf = vpCandidateKFs[idx];
        if (!pkf || pkf->isBad()) {
            vbDiscard[idx] = true;
            continue;
        }
        ORBMatcher matcher(0.75, true);
        std::vector<cv::DMatch> matches;
        mpCurrFrame->setMapPointsNull();
        int nMatches = matcher.searchByBow(mpCurrFrame, pkf, matches);
        relocBowParam.mvAllMatches[idx] = matches;
        if (nMatches < 10)
            vbDiscard[idx] = true;
        else {
            std::vector<cv::Mat> mapPoints;
            std::vector<cv::KeyPoint> ORBPoints;
            const auto &allORBPoints = mpCurrFrame->getLeftKeyPoints();
            const auto &allMapPoints = pkf->getMapPoints();

            std::size_t pnpIdx = 0;
            for (std::size_t jdx = 0; jdx < matches.size(); ++jdx) {
                const auto &match = matches[jdx];
                const auto &pMp = allMapPoints[match.trainIdx];
                const auto &ORBPoint = allORBPoints[match.queryIdx];
                if (!pMp || pMp->isBad())
                    continue;
                ORBPoints.push_back(ORBPoint);
                mapPoints.push_back(pMp->getPos().clone());
                relocBowParam.mvPnPId2MatchID[idx][pnpIdx] = jdx;
                ++pnpIdx;
            }
            relocBowParam.mvpSolvers[idx] = PnPSolver::create(mapPoints, ORBPoints);
            ++nCandidates;
        }
    }
    return nCandidates;
}

/**
 * @brief 设置当前帧的地图点
 *
 * @param vInliers          输入的RANSAC算法后，内点的分布位置（EPNP）
 * @param relocBowParam     输入的基于词袋匹配的信息
 * @param idx               输入的候选关键帧的索引
 * @param vpCandidateKFs    输入的候选关键帧
 */
void Tracking::setCurrFrameAttrib(const std::vector<std::size_t> &vInliers, const RelocBowParam &relocBowParam,
                                  const std::size_t &idx, std::vector<KeyFrame::SharedPtr> &vpCandidateKFs,
                                  const cv::Mat &Rcw, const cv::Mat &tcw) {
    mpCurrFrame->setMapPointsNull();
    for (const std::size_t &inlierIdx : vInliers) {
        std::size_t matchID = relocBowParam.mvPnPId2MatchID[idx].at(inlierIdx);
        int pMpId = relocBowParam.mvAllMatches[idx][matchID].trainIdx;
        int kPId = relocBowParam.mvAllMatches[idx][matchID].queryIdx;
        MapPoint::SharedPtr pMp = vpCandidateKFs[idx]->getMapPoints()[pMpId];
        if (pMp && !pMp->isBad())
            mpCurrFrame->setMapPoint(kPId, pMp);
    }
    cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
    Rcw.copyTo(Tcw(cv::Range(0, 3), cv::Range(0, 3)));
    tcw.copyTo(Tcw(cv::Range(0, 3), cv::Range(3, 4)));
    mpCurrFrame->setPose(Tcw);
}

/**
 * @brief 当恒速模型跟踪和参考关键帧跟踪失败时，尝试重定位跟踪
 * @details
 *      1. 使用关键帧数据库，寻找一些和当前帧最相似的候选关键帧
 *      2. 使用词袋匹配（无先验信息），找到和候选关键帧的匹配
 *      3. 使用EPnP算法和RANSAC模型，进行位姿的初步估计
 *      4. 使用重投影匹配，进行精确匹配
 *      5. 使用仅位姿优化，得到精确位姿
 *
 * @return true     重定位跟踪成功
 * @return false    重定位跟踪失败
 */
bool Tracking::trackReLocalize() {
    /// 1. 使用关键帧数据库，找到初步的候选关键帧
    int candidateNum;
    std::vector<KeyFrame::SharedPtr> vpCandidateKFs;
    if (!findInitialKF(vpCandidateKFs, candidateNum))
        return false;

    /// 2. 使用词袋匹配进一步筛选候选关键帧
    std::vector<bool> vbDiscard(candidateNum, false);
    RelocBowParam relocBowParam(candidateNum);
    int nCandidates = filterKFByBow(relocBowParam, vbDiscard, candidateNum, vpCandidateKFs);

    /// 3. 使用RANSAC和EPnP获取当前帧的初步位姿和内点分布
    std::vector<std::size_t> vInliers;
    bool bContinue = true;
    while (nCandidates && bContinue) {
        for (std::size_t idx = 0; idx < candidateNum; ++idx) {
            if (vbDiscard[idx])
                continue;
            bool bNoMore = false;
            cv::Mat Rcwi, tcwi;
            vInliers.clear();
            bool ret = relocBowParam.mvpSolvers[idx]->iterate(5, Rcwi, tcwi, bNoMore, vInliers);
            if (bNoMore) {
                vbDiscard[idx] = true;
                --nCandidates;
            }
            if (!ret)
                continue;
            setCurrFrameAttrib(vInliers, relocBowParam, idx, vpCandidateKFs, Rcwi, tcwi);
            int nInliers = Optimizer::OptimizePoseOnly(mpCurrFrame);
            if (nInliers < 10)
                continue;
            else if (nInliers < 50) {
                /// 4. 使用重投影匹配进行精确匹配
                if (addMatchByProject(vpCandidateKFs[idx], nInliers)) {
                    bContinue = false;
                    break;
                }
            } else {
                bContinue = false;
                break;
            }
        }
    }
    if (bContinue)
        return false;
    return true;
}

/**
 * @brief 使用重投影匹配进行精确匹配（修正位姿 + 添加匹配）
 * @details
 *      1. 在词袋匹配小于50时，使用
 *      2. 如果通过第一次词袋匹配，mpCurrFrame没有50个匹配点，直接返回失败
 *      3. 进行仅位姿优化，如果仅位姿优化得到的内点少于50，使用更严格的方式再次匹配
 *      4. 这次不进行位姿优化，而是统计匹配数目和内点数目的和是否大于50
 * @param pKFrame   输入的候选关键帧
 * @param nInliers  输入输出的mpCurrframe中的内点数目
 * @return true     成功
 * @return false    失败
 */
bool Tracking::addMatchByProject(KeyFrame::SharedPtr pKFrame, int &nInliers) {
    if (!pKFrame || pKFrame->isBad())
        return false;
    ORBMatcher matcher2(0.9, true);
    std::vector<cv::DMatch> matches2;
    int nAddition = matcher2.searchByProjection(mpCurrFrame, pKFrame, matches2, 10);
    if (nAddition + nInliers < 50)
        return false;
    nInliers = Optimizer::OptimizePoseOnly(mpCurrFrame);
    if (nInliers < 50) {
        nAddition = matcher2.searchByProjection(mpCurrFrame, pKFrame, matches2, 3);
        if (nAddition + nInliers < 50)
            return false;
    }
    return true;
}

/**
 * @brief 跟踪局部地图
 * @details
 *      1. 构建局部地图
 *      2. 将局部地图点投影到当前关键帧中，获取匹配
 *      3. 进行仅位姿优化
 *      4. 需要更新当前关键帧的参考关键帧
 * @return true
 * @return false
 */
bool Tracking::trackLocalMap() {
    buildLocalMap();
    ORBMatcher matcher(0.7, true);
    float th = 3;
    // todo 这里考虑是否发生了重定位，如果短时间内发生了重定位，th变大
    int nMatches = matcher.searchByProjection(mpCurrFrame, mvpLocalMps, th);
    if (nMatches < 30)
        return false;
    int nInliers = Optimizer::OptimizePoseOnly(mpCurrFrame);
    if (nInliers < 30)
        return false;
    // todo 这里需要考虑，如果距离重定位较近，需要的内点就要多
    return true;
}

/**
 * @brief 处理上一帧
 * @details
 *      1. 利用参考关键帧进行上一帧的位姿纠正
 *      2. 对上一帧进行反投影，构造临时地图点
 *      3. 使用Tracking进行临时地图点的维护
 *      4. 值得注意的是，维护的临时地图点中，有nullptr
 */
void Tracking::processLastFrame() {
    auto refKf = mpLastFrame->getRefKF();
    assert(refKf && "上一帧的参考关键帧为空！");
    mpLastFrame->setPose(mTlr * refKf->getPose());

    std::vector<MapPoint::SharedPtr> mapPoints;
    mpLastFrame->unProject(mapPoints);
    std::swap(mapPoints, mvpTempMappoints);
}

} // namespace ORB_SLAM2_ROS2
