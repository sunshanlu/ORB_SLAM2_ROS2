#include <iostream>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/LocalMapping.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Optimizer.h"
#include "ORB_SLAM2/PnPSolver.h"
#include "ORB_SLAM2/Tracking.h"
#include "ORB_SLAM2/Viewer.h"

namespace ORB_SLAM2_ROS2
{

using namespace std::chrono_literals;

Tracking::Tracking(Map::SharedPtr pMap, KFrameDBPtr pKfDB, VocPtr pVoc, const std::string &sBriefTemplate, int nFeatures, int nInitFeatures, int nInitFAST,
                   int nMinFAST, int nDepthTh, int nMaxFrame, int nMinFrame, int nColorType, bool bOnlyTracking, float fDepthScale, int nLevel,
                   float fScaleFactor)
    : mpMap(pMap)
    , mpKfDB(pKfDB)
    , mpVoc(pVoc)
    , msBriefTemFp(sBriefTemplate)
    , mnFeatures(nFeatures)
    , mnInitFeatures(nInitFeatures)
    , mnMaxThresh(nInitFAST)
    , mnMinThresh(nMinFAST)
    , mnDepthTh(nDepthTh)
    , mnMaxFrames(nMaxFrame)
    , mnMinFrames(nMinFrame)
    , mnColorType(nColorType)
    , mbOnlyTracking(bOnlyTracking)
    , mbUpdate(false)
    , mnLastRelocId(-1)
    , mnLastInsertId(0)
    , mStatus(TrackingState::NOT_IMAGE_YET)
    , mfdScale(fDepthScale)
    , mnLevels(nLevel)
    , mfScaleFactor(fScaleFactor)
{
}

/**
 * @brief 处理普通帧
 *
 * @param leftImg   左图像
 * @param rightImg  右图像
 * @return cv::Mat 输出的位姿矩阵
 */
cv::Mat Tracking::grabFrame(cv::Mat leftImg, cv::Mat rightImg)
{
  mleftImg = leftImg;
  if (mnColorType == 1)
  {
    /// 对应的RGB图像
    cv::cvtColor(leftImg, leftImg, cv::COLOR_RGB2GRAY);
    if (Camera::mType == CameraType::Stereo)
      cv::cvtColor(rightImg, rightImg, cv::COLOR_RGB2GRAY);
  }
  else if (mnColorType == 2)
  {
    /// 对应的BGR图像
    cv::cvtColor(leftImg, leftImg, cv::COLOR_BGR2GRAY);
    if (Camera::mType == CameraType::Stereo)
      cv::cvtColor(rightImg, rightImg, cv::COLOR_BGR2GRAY);
  }
  // switch (mStatus) {
  // case TrackingState::NOT_IMAGE_YET:
  //     mpCurrFrame =
  //         Frame::createStereo(leftImg, rightImg, mnInitFeatures, msBriefTemFp, mnMaxThresh, mnMinThresh, mpVoc);
  //     break;
  // case TrackingState::NOT_INITING:
  //     mpCurrFrame =
  //         Frame::createStereo(leftImg, rightImg, mnInitFeatures, msBriefTemFp, mnMaxThresh, mnMinThresh, mpVoc);
  //     break;
  // default:
  //     mpCurrFrame = Frame::createStereo(leftImg, rightImg, mnFeatures, msBriefTemFp, mnMaxThresh, mnMinThresh,
  //     mpVoc);
  // }
  switch (Camera::mType)
  {
  case CameraType::Stereo:
    mpCurrFrame = Frame::createStereo(leftImg, rightImg, mnFeatures, msBriefTemFp, mnMaxThresh, mnMinThresh, mpVoc, mnLevels, mfScaleFactor);
    break;
  case CameraType::RGBD:
    mpCurrFrame = Frame::createRGBD(leftImg, rightImg, mnFeatures, msBriefTemFp, mnMaxThresh, mnMinThresh, mpVoc, mfdScale, mnLevels, mfScaleFactor);
    break;
  default:
    throw std::runtime_error("此ORB_SLAM2只能在双目或者RGBD图像中使用");
    break;
  }

  bool bOK = false;
  if (!mbOnlyTracking)
  {
    /// 初始化阶段
    if (mStatus == TrackingState::NOT_IMAGE_YET || mStatus == TrackingState::NOT_INITING)
    {
      mStatus = TrackingState::NOT_INITING;
      auto pose = cv::Mat::eye(4, 4, CV_32F);
      mpCurrFrame->setPose(pose);
      if (mpCurrFrame->getN() >= 500)
      {
        initForStereo();
        mpLastFrame = mpCurrFrame;
        if (mpViewer)
          mpViewer->setCurrFrame(mleftImg, mpCurrFrame, mStatus);
      }
      return mpCurrFrame->getPose();
    }
  }
  else
  {
    if (mStatus == TrackingState::NOT_IMAGE_YET)
      mStatus = TrackingState::LOST;
  }
  if (mStatus == TrackingState::OK)
  {
    mpCurrFrame->setPose(mpLastFrame->getPose());
    if (mbOnlyTracking || mVelocity.empty() || mpCurrFrame->getID() < mnLastRelocId + 2)
    {
      bOK = trackReference();
    }
    else
    {
      bOK = trackMotionModel();
      if (!bOK)
        bOK = trackReference();
    }
  }
  else
    bOK = trackReLocalize();
  if (bOK)
    bOK = trackLocalMap();

  if (bOK)
  {
    mStatus = TrackingState::OK;
    updateVelocity();
    updateTlr();
    if (mpLocalMapper && !mbOnlyTracking)
      insertKeyFrame();

    mpLastFrame = mpCurrFrame;
    if (mpViewer)
      mpViewer->setCurrFrame(mleftImg, mpCurrFrame, mStatus);
    if (mbOnlyTracking && mpRefKf && !mpRefKf->isBad())
    {
      mpRefKf->setNotErased(false);
      mpRefKf = mpMap->getTrackingRef(mpCurrFrame, mpRefKf->getID());
      mpRefKf->setNotErased(true);
    }
    return mpCurrFrame->getPose();
  }
  else
  {
    mStatus = TrackingState::LOST;
    if (mpViewer)
      mpViewer->setCurrFrame(mleftImg, mpCurrFrame, mStatus);
    return cv::Mat();
  }
}

/// 向局部建图线程中插入关键帧
void Tracking::insertKeyFrame()
{
  if (!needNewKeyFrame())
    return;
  auto kf = updateCurrFrame();
  kf->setNotErased(true);
  mpRefKf->setNotErased(false);
  mpRefKf = kf;
  mpLocalMapper->insertKeyFrame(kf);
  mnLastInsertId = mpCurrFrame->getID();
}

/// 将当前帧升级为关键帧
KeyFrame::SharedPtr Tracking::updateCurrFrame()
{
  std::vector<MapPoint::SharedPtr> mps;
  int nCreated = mpCurrFrame->unProject(mps);
  return KeyFrame::create(*mpCurrFrame);
}

/// 更新速度Tcl
void Tracking::updateVelocity()
{
  if (!mpLastFrame)
  {
    mVelocity = cv::Mat();
    return;
  }
  cv::Mat Tcw = mpCurrFrame->getPose();
  cv::Mat Twl = mpLastFrame->getPoseInv();
  mVelocity = Tcw * Twl;
}

/// 更新mTlr（上一帧到参考关键帧的位姿）
void Tracking::updateTlr()
{
  /// 因为只是用到参考关键帧的速度，因此这里没什么问题
  KeyFrame::SharedPtr refKF = mpCurrFrame->getRefKF();
  if (!refKF)
  {
    refKF = mpRefKf;
  }
  mTlr = mpCurrFrame->getPose() * refKF->getPoseInv();
}

/**
 * @brief 插入关键帧到局部地图中去
 * 插入到局部地图中的关键帧的mbIsLocalKf会被标注为true
 * 同时匹配相同关键的同时插入（判断是否可用）
 */
void Tracking::insertLocalKFrame(KeyFrame::SharedPtr pKf)
{
  if (pKf && !pKf->isBad())
  {
    auto pParent = pKf->getParent().lock();
    auto vpChildren = pKf->getChildren();
    if (pParent && !pParent->isBad())
    {
      if (!pParent->isLocalKf())
      {
        pParent->setLocalKf(true);
        mvpLocalKfs.push_back(pParent);
      }
    }
    for (auto &childWeak : vpChildren)
    {
      auto child = childWeak.lock();
      if (child && !child->isBad())
      {
        if (!child->isLocalKf())
        {
          child->setLocalKf(true);
          mvpLocalKfs.push_back(child);
        }
      }
    }
    if (pKf->isLocalKf())
    {
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
void Tracking::insertLocalMPoint(MapPoint::SharedPtr pMp)
{
  if (pMp && !pMp->isBad())
  {
    if (!pMp->isLocalMp())
    {
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
void Tracking::buildLocalKfs()
{
  for (auto &kf : mvpLocalKfs)
    kf->setLocalKf(false);
  mvpLocalKfs.clear();
  auto firstKfs = mpCurrFrame->getConnectedKfs(0);
  for (auto &kf : firstKfs)
  {
    insertLocalKFrame(kf);
    auto secondKfs = kf->getConnectedKfs(0);
    for (auto &kf2 : secondKfs)
    {
      insertLocalKFrame(kf2);
    }
  }
}

/**
 * @brief 构建局部地图中的地图点
 * 注意，在使用这个api之前，保证局部地图中的关键帧已经构建完毕
 */
void Tracking::buildLocalMps()
{
  {
    std::unique_lock<std::mutex> lock(mMutexLMps);
    for (auto &pMp : mvpLocalMps)
      pMp->setLocalMp(false);
    mvpLocalMps.clear();

    for (auto &kf : mvpLocalKfs)
    {
      auto vpMps = kf->getMapPoints();
      for (auto &pMp : vpMps)
        insertLocalMPoint(pMp);
    }
  }
  setUpdate(true);
}

/**
 * @brief 构建局部地图
 * @details
 *      1. 构建局部地图中的关键帧
 *      2. 构建局部地图中的地图点
 */
void Tracking::buildLocalMap()
{
  buildLocalKfs();
  buildLocalMps();
}

/**
 * @brief 双目相机的初始化
 *
 */
void Tracking::initForStereo()
{
  std::vector<MapPoint::SharedPtr> mapPoints;
  mpCurrFrame->unProject(mapPoints);
  auto kfInit = KeyFrame::create(*mpCurrFrame);

  mpMap->insertKeyFrame(kfInit, mpMap);
  for (std::size_t idx = 0; idx < mapPoints.size(); ++idx)
  {
    auto &pMp = mapPoints[idx];
    if (!pMp)
      continue;
    pMp->addAttriInit(kfInit, idx);
    mpMap->insertMapPoint(pMp, mpMap);
  }
  mStatus = TrackingState::OK;
  mpRefKf = kfInit;
  mpLocalMapper->insertKeyFrame(kfInit);
}

/**
 * @brief 跟踪参考关键帧
 * @details
 *      1. 基于词袋的匹配要求成功匹配大于等于15
 *      2. 基于基于OptimizePoseOnly的优化，要求内点数目大于等于10
 * @return true     跟踪参考关键帧失败
 * @return false    跟踪参考关键帧成功
 */
bool Tracking::trackReference()
{
  ORBMatcher matcher(0.7, true);
  std::vector<cv::DMatch> matches;
  int nMatches = matcher.searchByBow(mpCurrFrame, mpRefKf, matches);
  if (nMatches < 10)
  {
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
bool Tracking::trackMotionModel()
{
  std::vector<cv::DMatch> matches;
  processLastFrame();
  mpCurrFrame->setPose(mVelocity * mpLastFrame->getPose());
  ORBMatcher matcher(0.9, true);
  int nMatches = matcher.searchByProjection(mpCurrFrame, mpLastFrame, matches, 15);
  if (nMatches < 20)
  {
    nMatches += matcher.searchByProjection(mpCurrFrame, mpLastFrame, matches, 30);
  }
  if (nMatches < 20)
  {
    return false;
  }
  Optimizer::OptimizePoseOnly(mpCurrFrame);
  int inLiers = 0;
  auto vMps = mpCurrFrame->getMapPoints();
  for (std::size_t idx = 0; idx < vMps.size(); ++idx)
  {
    auto &pMp = vMps[idx];
    if (pMp && pMp->isInMap())
      ++inLiers;
  }
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
bool Tracking::findInitialKF(std::vector<KeyFrame::SharedPtr> &vpCandidateKFs, int &candidateNum)
{
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
RelocBowParam::RelocBowParam(int nCandidate)
{
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
                            std::vector<KeyFrame::SharedPtr> &vpCandidateKFs)
{
  int nCandidates = 0;
  for (std::size_t idx = 0; idx < candidateNum; ++idx)
  {
    KeyFrame::SharedPtr pkf = vpCandidateKFs[idx];
    if (!pkf || pkf->isBad())
    {
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
    else
    {
      std::vector<cv::Mat> mapPoints;
      std::vector<cv::KeyPoint> ORBPoints;
      const auto &allORBPoints = mpCurrFrame->getLeftKeyPoints();
      const auto &allMapPoints = pkf->getMapPoints();

      std::size_t pnpIdx = 0;
      for (std::size_t jdx = 0; jdx < matches.size(); ++jdx)
      {
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
void Tracking::setCurrFrameAttrib(const std::vector<std::size_t> &vInliers, const RelocBowParam &relocBowParam, const std::size_t &idx,
                                  std::vector<KeyFrame::SharedPtr> &vpCandidateKFs, const cv::Mat &Rcw, const cv::Mat &tcw)
{
  mpCurrFrame->setMapPointsNull();
  for (const std::size_t &inlierIdx : vInliers)
  {
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
bool Tracking::trackReLocalize()
{
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
  KeyFrame::SharedPtr pRelocKf = nullptr;
  while (nCandidates && bContinue)
  {
    for (std::size_t idx = 0; idx < candidateNum; ++idx)
    {
      if (vbDiscard[idx])
        continue;
      bool bNoMore = false;
      PnPRet modelReti;
      vInliers.clear();
      bool ret = relocBowParam.mvpSolvers[idx]->iterate(5, modelReti, bNoMore, vInliers);
      if (bNoMore)
      {
        vbDiscard[idx] = true;
        --nCandidates;
      }
      if (!ret)
        continue;
      setCurrFrameAttrib(vInliers, relocBowParam, idx, vpCandidateKFs, modelReti.mRcw, modelReti.mtcw);
      int nInliers = Optimizer::OptimizePoseOnly(mpCurrFrame);
      if (nInliers < 10)
        continue;
      else if (nInliers < 50)
      {
        /// 4. 使用重投影匹配进行精确匹配
        if (addMatchByProject(vpCandidateKFs[idx], nInliers))
        {
          pRelocKf = vpCandidateKFs[idx];
          bContinue = false;
          break;
        }
      }
      else
      {
        pRelocKf = vpCandidateKFs[idx];
        bContinue = false;
        break;
      }
    }
  }
  if (bContinue)
    return false;
  if (pRelocKf && !pRelocKf->isBad())
  {
    pRelocKf->setNotErased(true);
    if (mpRefKf && !mpRefKf->isBad())
      mpRefKf->setNotErased(false);
    mpRefKf = pRelocKf;
  }
  mnLastRelocId = mpCurrFrame->getID();
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
bool Tracking::addMatchByProject(KeyFrame::SharedPtr pKFrame, int &nInliers)
{
  if (!pKFrame || pKFrame->isBad())
    return false;
  ORBMatcher matcher2(0.9, true);
  std::vector<cv::DMatch> matches2;
  int nAddition = matcher2.searchByProjection(mpCurrFrame, pKFrame, matches2, 10);
  if (nAddition + nInliers < 50)
    return false;
  nInliers = Optimizer::OptimizePoseOnly(mpCurrFrame);
  if (nInliers < 50)
  {
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
bool Tracking::trackLocalMap()
{
  buildLocalMap();
  ORBMatcher matcher(0.8, true);
  float th = 3;
  if (mnLastRelocId > 0)
    if (mpCurrFrame->getID() < mnLastRelocId + 2)
      th = 5;

  std::vector<cv::DMatch> matches;
  int nMatches = 0;
  {
    std::unique_lock<std::mutex> lock(mMutexLMps);
    nMatches = matcher.searchByProjection(mpCurrFrame, mvpLocalMps, th, matches);
  }
  if (nMatches < 30)
    return false;
  Optimizer::OptimizePoseOnly(mpCurrFrame);
  int nInliers = 0;
  std::size_t nNum = mpCurrFrame->getLeftKeyPoints().size();
  auto mps = mpCurrFrame->getMapPoints();
  for (std::size_t idx = 0; idx < nNum; ++idx)
  {
    auto &pMp = mps[idx];
    if (pMp && pMp->isInMap())
      ++nInliers;
    else if (pMp && !pMp->isInMap())
      mpCurrFrame->setMapPoint(idx, nullptr);
  }
  if (nInliers < 30)
    return false;
  if (mpCurrFrame->getID() < mnLastRelocId + mnMaxFrames && nInliers < 50)
    return false;
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
void Tracking::processLastFrame()
{
  auto refKf = mpLastFrame->getRefKF();
  assert(refKf && "上一帧的参考关键帧为空！");
  mpLastFrame->setPose(mTlr * refKf->getPose());

  std::vector<MapPoint::SharedPtr> mapPoints;
  mpLastFrame->unProject(mapPoints);
  std::swap(mapPoints, mvpTempMappoints);
}

/**
 * @brief 判断是否需要插入关键帧
 * @details
 *      1. 条件1，以下三个条件中，满足一个即可
 *          (1) 距离上次插入关键帧的时间超过1s，认为时间比较久了
 *          (2) 满足插入关键帧的最小帧数间隔且局部建图线程有空
 *          (3) 在双目或者RGBD情况下，满足一个即可
 *              a) 跟踪到的点，不足参考关键帧地图点的1/4
 *              b) 跟踪到的近点少于没有跟踪到的近点
 *      2. 条件2，满足((1) || (2)) && (3)
 *          (1) 当前跟踪到的点数目，小于阈值比例
 *              a) 单目相机情况下是0.9
 *              b) 双目相机或者RGBD相机是0.75
 *              c) 地图中只有只有一个关键帧时是0.4
 *          (2) 同条件1的(3)部分条件
 *          (3) 成功跟踪到的匹配内点数目大于15
 *      3. 不需要插入关键帧的情况
 *          (1) 仅定位模式下
 *          (2) 局部建图线程被闭环线程使用
 *          (3) 如果距离上一次重定位比较近（1s以内）
 *          (4) 如果地图中关键帧数目超过最大限制
 *
 * @return true     局部建图线程需要插入关键帧
 * @return false    局部建图线程不需要插入关键帧
 */
bool Tracking::needNewKeyFrame()
{
  if (mbOnlyTracking)
    return false;

  if (mnLastRelocId > 0)
    if (mpCurrFrame->getID() - mnLastRelocId <= mnMaxFrames)
      return false;
  if (mpLocalMapper->isRequestStop())
  {
    return false;
  }

  auto refMps = mpRefKf->getMapPoints();
  auto currMps = mpCurrFrame->getMapPoints();

  int nRefMps = 0, nCurrMps = 0;
  int nObs = mpRefKf->getID() == 0 ? 0 : 1;
  for (std::size_t idx = 0; idx < refMps.size(); ++idx)
  {
    auto &pMp = refMps[idx];
    if (pMp && !pMp->isBad() && pMp->getObsNum() > nObs)
      ++nRefMps;
    else
      mpRefKf->setMapPoint(idx, nullptr);
  }
  for (std::size_t idx = 0; idx < currMps.size(); ++idx)
  {
    auto &pMp = currMps[idx];
    if (pMp && !pMp->isBad() && pMp->isInMap())
      ++nCurrMps;
    else
    {
      mpCurrFrame->setMapPoint(idx, nullptr);
      currMps[idx] = nullptr;
    }
  }
  auto &vDepths = mpCurrFrame->getDepth();
  double depthTh = Camera::mfBl * mnDepthTh;

  int nTrackedClose = 0, nNoTrackedClose = 0;
  for (std::size_t idx = 0; idx < vDepths.size(); ++idx)
  {
    const double &depth = vDepths[idx];
    auto &pMp = currMps[idx];
    if (depth < depthTh && depth > 0)
    {
      if (pMp && !pMp->isBad())
        ++nTrackedClose;
      else
        ++nNoTrackedClose;
    }
  }
  float ratio = nCurrMps / ((float)nRefMps + 1e-5);

  bool bNeedClose = nTrackedClose < 100 && nNoTrackedClose > 70;
  bool bLocalMapperIdle = mpLocalMapper->getAcceptKF();
  bool c1a = mpCurrFrame->getID() - mnLastInsertId > mnMaxFrames;
  bool c1b = mpCurrFrame->getID() - mnLastInsertId > mnMinFrames && bLocalMapperIdle;
  bool c1c = ratio < 0.25 || bNeedClose;
  bool c1 = c1a || c1b || c1c;

  float ratioTh = 0.75;
  if (mpMap->keyFramesInMap() < 2)
    ratioTh = 0.4;
  bool c2a = ratio < ratioTh;
  bool c2b = bNeedClose;
  bool c2 = c2a || c2b;
  bool flag1 = c1 && c2;
  if (!flag1)
    return false;
  if (bLocalMapperIdle)
    return true;
  else
  {
    mpLocalMapper->setAbortBA(true);
    if (mpLocalMapper->getKFNum() < 3)
    {
      return true;
    }
    else
      return false;
  }
}

} // namespace ORB_SLAM2_ROS2
