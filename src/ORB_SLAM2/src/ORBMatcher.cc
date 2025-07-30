#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/Sim3Solver.h"

namespace ORB_SLAM2_ROS2
{

/**
 * @brief 双目相机初始化匹配
 *      1. 利用双目极线进行粗匹配
 *      2. 利用SAD+二次项差值进行精确匹配
 * @param pFrame    寻找匹配的帧
 * @return int      返回匹配的数目
 */
int ORBMatcher::searchByStereo(Frame::SharedPtr pFrame)
{
  int nMatches = 0;
  int nLeftKp = pFrame->mvFeatsLeft.size();
  pFrame->mvDepths.clear();
  pFrame->mvFeatsRightU.clear();
  pFrame->mvDepths.resize(nLeftKp, -1.0);
  pFrame->mvFeatsRightU.resize(nLeftKp, -1.0);

  const auto &leftPyramids = pFrame->getLeftPyramid();
  const auto &rightPyramids = pFrame->getRightPyramid();
  const auto &leftKeyPoints = pFrame->getLeftKeyPoints();
  const auto &rightKeyPoints = pFrame->getRightKeyPoints();
  const auto &leftDesc = pFrame->getLeftDescriptor();
  const auto &rightDesc = pFrame->getRightDescriptor();

  RowIdxDB rowIdxDB = ORBMatcher::createRowIndexDB(pFrame.get());
  for (std::size_t ldx = 0; ldx < leftKeyPoints.size(); ++ldx)
  {
    const auto &lKp = leftKeyPoints[ldx];
    float maxU = lKp.pt.x - 0;
    float minU = std::max(0.f, lKp.pt.x - Camera::mfFx);
    cv::Mat lDesc = leftDesc.at(ldx);
    const auto &rKpIds = rowIdxDB[cvRound(lKp.pt.y)];
    std::vector<std::size_t> candidateIdx;
    std::copy_if(rKpIds.begin(), rKpIds.end(), std::back_inserter(candidateIdx),
                 [&](const std::size_t &idx)
                 {
                   const float &retCol = rightKeyPoints[idx].pt.x;
                   return (retCol < maxU && retCol > minU) ? true : false;
                 });
    if (candidateIdx.empty())
      continue;
    float ratio = 0.f;
    BestMatchDesc bestMatch = ORBMatcher::getBestMatch(lDesc, rightDesc, candidateIdx, ratio);
    if (bestMatch.second > mnMeanThreshold)
      continue;

    const auto &rKp = rightKeyPoints[bestMatch.first];
    if (lKp.octave > rKp.octave + 1 || lKp.octave < rKp.octave - 1)
      continue;

    const cv::Mat &leftImage = leftPyramids[lKp.octave];
    const cv::Mat &rightImage = rightPyramids[rKp.octave];
    float deltaU = pixelSADMatch(leftImage, rightImage, lKp, rKp);

    float rightU = rKp.pt.x + deltaU;
    rightU = std::max(0.f, rightU);
    rightU = std::min(rightU, (float)rightPyramids[0].cols - 1);
    float delta = lKp.pt.x - rightU;
    if (delta <= 0)
    {
      rightU = rKp.pt.x;
      delta = lKp.pt.x - rightU;
      if (delta <= 0)
        continue;
    }
    pFrame->mvFeatsRightU[ldx] = rightU;

    pFrame->mvDepths[ldx] = Camera::mfBf / (lKp.pt.x - rightU);
    ++nMatches;
  }
  return nMatches;
}

/**
 * @brief 通过词袋加速匹配（不会丢弃恒速模型跟踪的地图点，更鲁邦）
 * @details
 *      1. 通过词袋获得pFrame和pKframe的匹配
 *      2. pFrame匹配成功的部分，跳过
 * @param pFrame    寻找匹配的普通帧
 * @param pKframe   参与匹配的关键帧
 * @param bAddMPs   是否添加新的地图点(双方都没有地图点)
 * @return int      输出的匹配数目
 */
// int ORBMatcher::searchByBow(VirtualFrame::SharedPtr pFrame, VirtualFrame::SharedPtr pKframe,
//                             std::vector<cv::DMatch> &matches, bool bAddMPs) {

//     pFrame->computeBow();
//     pKframe->computeBow();
//     auto pfI = pFrame->mFeatVec.begin();
//     auto pkfI = pKframe->mFeatVec.begin();
//     auto pfE = pFrame->mFeatVec.end();
//     auto pkfE = pKframe->mFeatVec.end();

//     int nMatch = 0;
//     std::vector<std::pair<int, float>> db;
//     auto mapPointsF = pFrame->getMapPoints();
//     auto mapPointsKF = pKframe->getMapPoints();

//     while (pfI != pfE && pkfI != pkfE) {
//         if (pfI->first > pkfI->first)
//             ++pkfI;
//         else if (pfI->first < pkfI->first)
//             ++pfI;
//         else {
//             for (const auto &pId : pfI->second) {
//                 const auto &pMpFrame = mapPointsF[pId];
//                 if (!bAddMPs) {
//                     if (pMpFrame && !pMpFrame->isBad())
//                         continue;
//                 } else {
//                     if (pMpFrame && !pMpFrame->isBad() && pMpFrame->isInMap())
//                         continue;
//                 }
//                 const auto &fDesc = pFrame->mvLeftDescriptor.at(pId);
//                 std::vector<std::size_t> candidateIds;
//                 for (const auto &pkId : pkfI->second) {
//                     const auto &pMp = mapPointsKF[pkId];
//                     bool goodFlag = (pMp && !pMp->isBad());
//                     if (!bAddMPs) {
//                         if (goodFlag) {
//                             candidateIds.push_back(pkId);
//                         }
//                     } else {
//                         if (!goodFlag || (pMp && !pMp->isInMap()))
//                             candidateIds.push_back(pkId);
//                     }
//                 }
//                 if (candidateIds.empty())
//                     continue;
//                 float ratio = 0.f;
//                 auto bestMatch = getBestMatch(fDesc, pKframe->mvLeftDescriptor, candidateIds, ratio);
//                 ++nMatch;
//                 if (bestMatch.second > mnMinThreshold || ratio > mfRatio) {
//                     db.push_back({bestMatch.second, ratio});
//                     continue;
//                 }
//                 matches.emplace_back(pId, bestMatch.first, bestMatch.second);
//             }
//             ++pfI;
//             ++pkfI;
//         }
//     }
//     if (mbCheckOri) {
//         verifyAngle(matches, pFrame->mvFeatsLeft, pKframe->getLeftKeyPoints());
//     }
//     if (!bAddMPs)
//         setMapPoints(pFrame->mvpMapPoints, mapPointsKF, matches);
//     return matches.size();
// }

/**
 * @brief 通过词袋加速匹配（不会丢弃恒速模型跟踪的地图点，更鲁邦）
 * @details
 *      1. 通过词袋获得pFrame和pKframe的匹配
 *      2. pFrame匹配成功的部分，跳过
 * @param pFrame    寻找匹配的普通帧
 * @param pKframe   参与匹配的关键帧
 * @param bAddMPs   是否添加新的地图点(双方都没有地图点)
 * @return int      输出的匹配数目
 */
int ORBMatcher::searchByBow(VirtualFrame::SharedPtr pFrame, VirtualFrame::SharedPtr pKframe, std::vector<cv::DMatch> &matches, bool bAddMPs, bool bLoop)
{
  pFrame->computeBow();
  pKframe->computeBow();
  auto pfI = pFrame->mFeatVec.begin();
  auto pkfI = pKframe->mFeatVec.begin();
  auto pfE = pFrame->mFeatVec.end();
  auto pkfE = pKframe->mFeatVec.end();
  int nMatch = 0;
  std::vector<std::pair<int, float>> db;
  auto mapPointsF = pFrame->getMapPoints();
  auto mapPointsKF = pKframe->getMapPoints();

  while (pfI != pfE && pkfI != pkfE)
  {
    if (pfI->first > pkfI->first)
      ++pkfI;
    else if (pfI->first < pkfI->first)
      ++pfI;
    else
    {
      for (const auto &pkId : pkfI->second)
      {
        const auto &pMpKFrame = mapPointsKF[pkId];
        bool goodFlag = (pMpKFrame && !pMpKFrame->isBad());

        /// 注意，bLoop和bAddMPs不同时为true
        if (bAddMPs)
        {
          if (goodFlag && pMpKFrame->isInMap())
            continue;
        }
        else if (bLoop)
        {
        }
        else
        {
          if (!goodFlag)
            continue;
        }
        const auto &fKDesc = pKframe->mvLeftDescriptor.at(pkId);
        std::vector<std::size_t> candidateIds;
        for (const auto &pId : pfI->second)
        {
          const auto &pMpFrame = mapPointsF[pId];
          bool goodFlag = (pMpFrame && !pMpFrame->isBad());
          if (bAddMPs)
          {
            if (goodFlag && pMpFrame->isInMap())
              continue;
            candidateIds.push_back(pId);
          }
          else if (bLoop)
          {
            candidateIds.push_back(pId);
          }
          else
          {
            if (!goodFlag)
              candidateIds.push_back(pId);
          }
        }
        if (candidateIds.empty())
          continue;
        float ratio = 0.f;
        auto bestMatch = getBestMatch(fKDesc, pFrame->mvLeftDescriptor, candidateIds, ratio);
        ++nMatch;
        if (bestMatch.second > mnMinThreshold || ratio > mfRatio)
        {
          db.push_back({bestMatch.second, ratio});
          continue;
        }
        matches.emplace_back(bestMatch.first, pkId, bestMatch.second);
      }
      ++pfI;
      ++pkfI;
    }
  }
  if (mbCheckOri)
    verifyAngle(matches, pFrame->getLeftKeyPoints(), pKframe->getLeftKeyPoints());
  if (!bAddMPs && !bLoop)
    setMapPoints(pFrame->mvpMapPoints, mapPointsKF, matches);
  return matches.size();
}

/**
 * @brief 恒速模型跟踪中的重投影匹配方法
 * @details
 *      1. 在重投影匹配前，需要根据恒速模型确定pFrame1的位姿
 *      2. 与ORB-SLAM2不同的是，这里没有对地图点进行筛选（因为位姿并不准确，为了增加跟踪鲁棒性）
 * @param pFrame1 输入的待匹配的帧
 * @param pFrame2 输入的参与匹配的帧
 * @param th      输入的寻找匹配的阈值大小
 * @return int  输出的匹配成功的个数
 */
int ORBMatcher::searchByProjection(VirtualFrame::SharedPtr pFrame1, VirtualFrame::SharedPtr pFrame2, std::vector<cv::DMatch> &matches, float th, bool bFuse)
{
  matches.clear();

  /// 判断前进或后退
  cv::Mat Rcw1, tcw1, Rcw2, tcw2;
  pFrame1->getPose(Rcw1, tcw1);
  pFrame2->getPose(Rcw2, tcw2);

  cv::Mat twc1 = -Rcw1.t() * tcw1;
  cv::Mat tlc = Rcw2 * twc1 + tcw2;
  float z = tlc.at<float>(2, 0);
  float zabs = std::abs(z);
  bool up = false, down = false;
  if (zabs > Camera::mfBl)
    z > 0 ? up = true : down = true;
  auto mps1 = pFrame1->getMapPoints();
  auto mps2 = pFrame2->getMapPoints();
  for (std::size_t idx = 0; idx < mps2.size(); ++idx)
  {
    MapPoint::SharedPtr pMp2 = mps2[idx];
    if (!pMp2 || pMp2->isBad())
      continue;
    if (bFuse)
    {
      float vecDistance, cosTheta;
      cv::Point2f uv;
      if (!pMp2->isInVision(pFrame1, vecDistance, uv, cosTheta))
      {
        continue;
      }
    }
    std::vector<std::size_t> candidateIdx;
    const auto &feature = pFrame2->mvFeatsLeft[idx];
    const auto &desc = pFrame2->mvLeftDescriptor[idx];
    if (up)
    {
      candidateIdx = pFrame1->findFeaturesInArea(feature, th, feature.octave, 7);
    }
    else if (down)
    {
      candidateIdx = pFrame1->findFeaturesInArea(feature, th, 0, feature.octave);
    }
    else
    {
      int minOctave = std::max(0, feature.octave - 1);
      int maxOctave = std::min(feature.octave + 1, 7);
      candidateIdx = pFrame1->findFeaturesInArea(feature, th, minOctave, maxOctave);
    }
    if (candidateIdx.empty())
    {
      continue;
    }
    std::vector<std::size_t> newCandidates;

    /// 如果是恒速模型匹配，将fram原有的匹配保留下来，不做任何修改
    if (!bFuse)
      std::copy_if(candidateIdx.begin(), candidateIdx.end(), std::back_inserter(newCandidates),
                   [&](const std::size_t &candidateID)
                   {
                     MapPoint::SharedPtr pMp1 = mps1[candidateID];
                     if (pMp1 && !pMp1->isBad())
                     {
                       pMp1->addMatchInTrack();
                       return false;
                     }
                     return true;
                   });
    else
      std::swap(candidateIdx, newCandidates);
    float ratio = 0.f;
    if (newCandidates.empty())
      continue;
    auto bestMatches = getBestMatch(desc, pFrame1->mvLeftDescriptor, newCandidates, ratio);
    if (ratio < mfRatio && bestMatches.second < mnMinThreshold)
    {
      matches.emplace_back(bestMatches.first, idx, bestMatches.second);
    }
  }
  if (!bFuse)
    setMapPoints(pFrame1->mvpMapPoints, pFrame2->mvpMapPoints, matches);
  return matches.size();
}

/**
 * @brief 基于SIM3相似性变换矩阵的投影
 * @details
 *      1. 将pMpC地图点投影到M坐标系中（mpCurr的位姿 + Smc）
 *      2. 判断地图点投影是否合格（mpCurr）
 *      3. 寻找在M坐标系中的匹配3D点索引（mpMatch + th + pM）
 *      4. 寻找最佳的匹配信息（desc + descM）
 * @param pMpC          输入的C坐标系下的地图点
 * @param Rcw           输入的C坐标系下的Rcw
 * @param tcw           输入的C坐标系下的tcw
 * @param g2oSmc        输入的Smc相似性变换矩阵
 * @param mpCurr        输入的关键帧C
 * @param mpMatch       输入的关键帧M
 * @param th            输入的搜索窗口阈值
 * @param pM            输入的M坐标系下的所有地图点
 * @param desc          输入的pMpC对应2d关键点的描述子
 * @param descM         输入的M坐标系下的所有关键点描述子
 * @param candidateID   输出的候选地图点在M坐标系下的索引
 * @return true     pMpC成功进行了匹配
 * @return false    pMpC匹配失败
 */
bool ORBMatcher::SIM3Project(MapPointPtr pMpC, const cv::Mat &Rcw, const cv::Mat &tcw, const Sim3Ret &g2oSmc, KeyFramePtr mpCurr, KeyFramePtr mpMatch,
                             const float &th, const std::vector<MapPointPtr> &pM, const cv::Mat &desc, const std::vector<cv::Mat> &descM,
                             std::size_t &candidateID)
{
  cv::Mat p3dW = pMpC->getPos();
  cv::Mat p3dC = Rcw * p3dW + tcw;
  cv::Mat p3dM = g2oSmc * p3dC;

  if (p3dM.at<float>(2) <= 0)
    return false;
  cv::KeyPoint kp;
  Camera::project(p3dM, kp.pt);
  if (!mpCurr->isInImage(kp.pt))
    return false;
  const float &x = p3dM.at<float>(0);
  const float &y = p3dM.at<float>(1);
  const float &z = p3dM.at<float>(2);
  // note: 这里需要同步到C坐标系下的尺度，因为是使用C坐标系下的点进行预测层级
  float d = std::sqrt(x * x + y * y + z * z) / g2oSmc.mfS;
  if (!pMpC->isGoodDistance(d))
    return false;
  kp.octave = pMpC->predictLevel(d);
  auto vIndices = mpMatch->findFeaturesInArea(kp, th, kp.octave - 1, kp.octave + 1);
  std::vector<std::size_t> vGoodIndices;
  std::copy_if(vIndices.begin(), vIndices.end(), std::back_inserter(vGoodIndices),
               [&](const std::size_t &id)
               {
                 auto &pMpM = pM[id];
                 if (pMpM && !pMpM->isBad() && pMpM->isInMap())
                   return true;
                 return false;
               });
  if (vGoodIndices.empty())
    return false;
  float ratio = 0;
  auto bestMatch = getBestMatch(desc, descM, vGoodIndices, ratio);
  if (bestMatch.second <= mnMinThreshold && ratio <= mfRatio)
  {
    candidateID = bestMatch.first;
    return true;
  }
  return false;
}

/**
 * @brief 回环闭合线程中的SIM3投影匹配
 *
 * @param mpCurr    回环闭合线程中的当前帧
 * @param mpMatch   回环闭合线程中的匹配帧
 * @param matches   SIM3Solver的内点匹配+新增匹配
 * @param g2oScm    输入的SIM3Solver计算的闭式解
 * @param th        输入的投影匹配的窗口阈值
 * @return int      输出的匹配点的数量（原有+新增的）
 */
int ORBMatcher::searchBySim3(KeyFramePtr mpCurr, KeyFramePtr mpMatch, std::vector<cv::DMatch> &matches, Sim3Ret &g2oScm, float th)
{
  auto pC = mpCurr->getMapPoints();
  auto pM = mpMatch->getMapPoints();
  auto &descC = mpCurr->getLeftDescriptor();
  auto &descM = mpMatch->getLeftDescriptor();
  int nc = pC.size(), nm = pM.size();
  std::vector<bool> goodflagC(nc, true);
  std::vector<bool> goodflagM(nm, true);
  for (const auto &match : matches)
  {
    goodflagC[match.queryIdx] = false;
    goodflagM[match.trainIdx] = false;
  }

  Sim3Ret g2oSmc = g2oScm.inv();
  cv::Mat Rcw, tcw, Rmw, tmw;
  mpCurr->getPose(Rcw, tcw);
  mpMatch->getPose(Rmw, tmw);

  std::map<std::size_t, std::size_t> mnewMatch;
  /// 反向投影，将C的地图点投影到M中去
  for (int ic = 0; ic < nc; ++ic)
  {
    if (!goodflagC[ic])
      continue;
    auto &pMpC = pC[ic];
    if (!pMpC || pMpC->isBad())
      continue;
    if (!pMpC->isInMap())
      continue;
    std::size_t bestID;
    int bestScore;
    bool ret = SIM3Project(pMpC, Rcw, tcw, g2oSmc, mpCurr, mpMatch, th, pM, descC[ic], descM, bestID);
    if (!ret)
      continue;
    mnewMatch.insert({ic, bestID});
  }
  /// 正向投影，将M的地图点投影到C中去
  for (int im = 0; im < nm; ++im)
  {
    if (!goodflagM[im])
      continue;
    auto &pMpM = pM[im];
    if (!pMpM || pMpM->isBad())
      continue;
    std::size_t bestID;
    bool ret = SIM3Project(pMpM, Rmw, tmw, g2oScm, mpMatch, mpCurr, th, pC, descM[im], descC, bestID);
    if (!ret)
      continue;
    mnewMatch.insert({bestID, im});
  }
  for (const auto &m : mnewMatch)
  {
    cv::DMatch match;
    match.queryIdx = m.first;
    match.trainIdx = m.second;
    matches.push_back(match);
  }
  return matches.size();
}

/**
 * @brief 将sLoopGroupMps投影到pCurr中，进行匹配
 * @details
 *      1. sLoopGroupMps中，如果已经参与匹配的，即vMatchedMps中存在的，跳过
 *      2. 投影时，需要注意，将Pw投影到C坐标系中时，尺度会乘以scw倍
 *          1) 在预测金字塔层级的时，尺度应该是世界坐标系下的尺度，缩小scw倍
 *          2) 在进行角度判断时，尺度并不会对角度产生影响，两向量之间的角度依然不变
 *      3. 这里投影的目的是后续的地图点的融合，即替换或新增，因此不需要必须匹配同时具有地图点的部分
 * @param pCurr         输入的回环闭合线程中的当前关键帧
 * @param vLoopGroupMps 输入的回环闭合线程中的回环闭合关键帧组中的地图点
 * @param vMatchedMps   输入输出的回环闭合线程中的产生匹配的地图点（已有的+新增的）
 * @param g2oScw        输入的g2o优化过的SIM3变换矩阵
 * @param th            输入的窗口搜索的阈值
 * @return int  输出的匹配数量（已有的 + 新增的）
 */
int ORBMatcher::searchBySim3(KeyFramePtr pCurr, const std::vector<MapPointPtr> &vLoopGroupMps, std::vector<MapPointPtr> &vMatchedMps, Sim3Ret &g2oScw, float th)
{
  int nMatches = 0;
  std::set<MapPoint::SharedPtr> sAlreadyMatched;
  for (const auto &pMpM : vMatchedMps)
  {
    if (pMpM && !pMpM->isBad() && pMpM->isInMap())
    {
      sAlreadyMatched.insert(pMpM);
      ++nMatches;
    }
  }

  auto alReadyEnd = sAlreadyMatched.end();
  for (auto &pMp : vLoopGroupMps)
  {
    if (!pMp || pMp->isBad() || !pMp->isInMap())
      continue;
    if (sAlreadyMatched.find(pMp) != alReadyEnd)
      continue;
    auto p3dW = pMp->getPos();
    auto p3dC = g2oScw * p3dW;
    if (p3dC.at<float>(2) <= 0)
      continue;
    cv::KeyPoint kp;
    Camera::project(p3dC, kp.pt);
    if (!pCurr->isInImage(kp.pt))
      continue;
    float dWithS = cv::norm(p3dC);
    float d = dWithS / g2oScw.mfS;
    if (!pMp->isGoodDistance(d))
      continue;
    cv::Mat viewDirection = pMp->getViewDirection();
    if ((g2oScw.mRqp * viewDirection).dot(p3dC) < 0.5 * dWithS)
      continue;
    kp.octave = pMp->predictLevel(d);
    auto vIndices = pCurr->findFeaturesInArea(kp, th, kp.octave - 1, kp.octave + 1);
    if (vIndices.empty())
      continue;
    float ratio = 0.f;
    auto bestMatch = getBestMatch(pMp->getDesc(), pCurr->getLeftDescriptor(), vIndices, ratio);
    if (bestMatch.second <= mnMinThreshold && ratio <= mfRatio)
    {
      vMatchedMps[bestMatch.first] = pMp;
      ++nMatches;
    }
  }
  return nMatches;
}

/**
 * @brief 跟踪局部地图中使用的重投影方法
 * @details
 *      1. 筛选半径需要考虑观测的方向，cos(theta) > 0.998，半径为2.5
 *      2. 筛选半径需要考虑金字塔层级，以金字塔的缩放因子作为标准差
 * @param pframe    待匹配的普通帧
 * @param mapPoints 参与匹配的局部地图中的地图点
 * @param th        最终产生的搜索半径后，需要乘的倍数
 * @return int      输出产生匹配的个数（新增的 + 已有的）
 */
int ORBMatcher::searchByProjection(VirtualFrame::SharedPtr pframe, const std::vector<MapPoint::SharedPtr> &mapPoints, float th,
                                   std::vector<cv::DMatch> &matches, bool bFuse)
{
  int nMatches = 0;
  auto pFrameMapPoints = pframe->getMapPoints();
  if (!bFuse)
    std::for_each(pFrameMapPoints.begin(), pFrameMapPoints.end(),
                  [&](MapPoint::SharedPtr pMp)
                  {
                    if (pMp && !pMp->isBad())
                      ++nMatches;
                  });
  for (std::size_t idx = 0; idx < mapPoints.size(); ++idx)
  {
    auto pMp = mapPoints[idx];
    if (!pMp || pMp->isBad() || !pMp->isInMap())
      continue;
    float distance, cosTheta;
    cv::KeyPoint kp;
    if (!pMp->isInVision(pframe, distance, kp.pt, cosTheta))
      continue;
    kp.octave = pMp->predictLevel(distance);
    float radius = cosTheta > 0.998f ? 2.5f : 4.0f;
    int minLevel = std::max(0, kp.octave - 1);
    int maxLevel = std::min(ORBExtractor::mnLevels - 1, kp.octave + 1);
    auto candidateIdx = pframe->findFeaturesInArea(kp, radius * th, minLevel, maxLevel);
    if (candidateIdx.empty())
      continue;
    float fRatio;
    auto bestMatch = getBestMatch(pMp->getDesc(), pframe->getLeftDescriptor(), candidateIdx, fRatio);
    if (bestMatch.second < mnMinThreshold && fRatio < mfRatio)
    {
      if (!bFuse)
      {
        auto pMpInF = pframe->getMapPoint(bestMatch.first);
        if (!pMpInF || pMpInF->isBad() || !pMpInF->isInMap())
        {
          pframe->setMapPoint(bestMatch.first, pMp);
          pMp->addMatchInTrack();
          ++nMatches;
        }
      }
      else
      {
        cv::DMatch match(bestMatch.first, idx, bestMatch.second);
        matches.push_back(match);
        ++nMatches;
      }
    }
  }
  return nMatches;
}

/**
 * @brief 处理融合的地图点
 *
 * @param matches       输入的匹配信息
 * @param fMapPoints    输入的被匹配的关键帧的地图点信息
 * @param vMapPoints    输入的参与匹配的地图点信息
 * @param pkf1          输入输出的被匹配的关键帧
 * @param map           输入输出的地图
 */
int ORBMatcher::processFuseMps(const std::vector<cv::DMatch> &matches, std::vector<MapPointPtr> &fMapPoints, std::vector<MapPointPtr> &vMapPoints,
                               KeyFramePtr &pkf1, MapPtr &map, bool bLoop)
{
  int nFuse = 0;
  for (auto &match : matches)
  {
    MapPoint::SharedPtr &pMp1 = fMapPoints[match.queryIdx];
    MapPoint::SharedPtr &pMp2 = vMapPoints[match.trainIdx];
    if (!pMp2 || pMp2->isBad())
      continue;
    if (!pMp1 || pMp1->isBad())
    {
      pkf1->setMapPoint(match.queryIdx, pMp2);
      pMp2->addObservation(pkf1, match.queryIdx);
      ++nFuse;
    }
    else
    {
      if (pMp1 == pMp2)
        continue;
      if (bLoop)
      {
        MapPoint::replace(pMp2, pMp1, map);
        ++nFuse;
      }
      else
      {
        int obs1 = pMp1->getObsNum();
        int obs2 = pMp2->getObsNum();
        if (obs1 >= obs2)
        {
          MapPoint::replace(pMp1, pMp2, map);
        }
        else
          MapPoint::replace(pMp2, pMp1, map);
        ++nFuse;
      }
    }
  }
  return nFuse;
}

/**
 * @brief 正向投影融合
 * @details
 *      1. 在投影之前，进行地图点的筛选操作
 *          1) 在相机的前方
 *          2) 投影之后再图像的范围内
 *          3) 距离要满足要求
 *          4) 观测角度要小于60度
 *          5) 对已经产生匹配的地图点进行剔除
 *      2. 使用重投影的方式进行投影匹配
 *          1) 对新增的投影匹配进行新增
 *          2) 对已有的投影匹配进行融合（保留匹配多的情况）
 * @param pkf1      输入的关键帧
 * @param mapPoints 输入的参与投影的地图点
 * @param map       输入的地图
 * @return int  成功融合的地图点数目（新增+替换）
 */
int ORBMatcher::fuse(KeyFrame::SharedPtr pkf1, const std::vector<MapPoint::SharedPtr> &mapPoints, MapPtr map, bool bLoop, float th)
{
  std::vector<cv::DMatch> matches;
  std::set<MapPoint::SharedPtr> sMapPoints;
  std::vector<MapPoint::SharedPtr> vMapPoints;
  auto fMapPoints = pkf1->getMapPoints();
  for (auto &pMp : fMapPoints)
  {
    if (pMp && !pMp->isBad())
    {
      sMapPoints.insert(pMp);
    }
  }
  for (auto &pMp : mapPoints)
  {
    /// 其他的筛选在searchByProjection都有
    if (sMapPoints.find(pMp) != sMapPoints.end())
      continue;
    vMapPoints.push_back(pMp);
  }
  int nMatches = searchByProjection(pkf1, vMapPoints, th, matches, true);
  fMapPoints = pkf1->getMapPoints();
  int nFuse = processFuseMps(matches, fMapPoints, vMapPoints, pkf1, map, bLoop);
  return nFuse;
}

/**
 * @brief 反向投影融合，将pkf2的地图点投影到pkf1中进行融合
 *
 * @param pkf1 输入的待融合的关键帧
 * @param pkf2 输入的参与融合的关键帧
 * @param map  输入的参与地图点变换的地图
 * @return int 输出的融合数目（新增+替换）
 */
int ORBMatcher::fuse(KeyFramePtr pkf1, KeyFramePtr pkf2, MapPtr map)
{
  std::vector<cv::DMatch> matches;
  int nMatches = searchByProjection(pkf1, pkf2, matches, 3.0f, true);
  auto fMapPoints = pkf1->getMapPoints();
  auto vMapPoints = pkf2->getMapPoints();
  int nFuse = processFuseMps(matches, fMapPoints, vMapPoints, pkf1, map);
  return nFuse;
}

/**
 * @brief 局部建图线程中，为三角化做准备
 * @details
 *      1. 使用词袋匹配，将两关键帧中都没有匹配的ORB特征点进行匹配
 *      2. 使用极线约束，进行剔除离群点 (相互极线约束)
 * @param pkf1      输入的待匹配的关键帧（当前关键帧）
 * @param pkf2      输入的参与匹配的关键帧（一阶相连关键帧）
 * @param matches   输出的匹配结果
 * @return int      输出的匹配成功的个数
 */
int ORBMatcher::searchForTriangulation(KeyFrame::SharedPtr pkf1, KeyFrame::SharedPtr pkf2, std::vector<cv::DMatch> &matches)
{
  int nAddMatches = searchByBow(pkf1, pkf2, matches, true);
  if (!nAddMatches)
    return 0;
  cv::Mat T21 = pkf2->getPose() * pkf1->getPoseInv();
  cv::Mat T12 = pkf1->getPose() * pkf2->getPoseInv();
  cv::Mat R21 = T21.rowRange(0, 3).colRange(0, 3);
  cv::Mat t21 = T21.rowRange(0, 3).colRange(3, 4);
  cv::Mat R12 = T12.rowRange(0, 3).colRange(0, 3);
  cv::Mat t12 = T12.rowRange(0, 3).colRange(3, 4);
  float x21 = t21.at<float>(0, 0);
  float y21 = t21.at<float>(1, 0);
  float z21 = t21.at<float>(2, 0);
  float x12 = t12.at<float>(0, 0);
  float y12 = t12.at<float>(1, 0);
  float z12 = t12.at<float>(2, 0);
  cv::Mat t21ssm = (cv::Mat_<float>(3, 3) << 0, -z21, y21, z21, 0, -x21, -y21, x21, 0);
  cv::Mat t12ssm = (cv::Mat_<float>(3, 3) << 0, -z12, y12, z12, 0, -x12, -y12, x12, 0);
  cv::Mat F21 = Camera::mKInv.t() * t21ssm * R21 * Camera::mKInv;
  cv::Mat F12 = Camera::mKInv.t() * t12ssm * R12 * Camera::mKInv;
  float th1 = 0, th2 = 0;
  std::vector<cv::DMatch> goodMatches;
  for (std::size_t idx = 0; idx < nAddMatches; ++idx)
  {
    const auto &match = matches[idx];
    const auto kpt1 = pkf1->mvFeatsLeft[match.queryIdx];
    const auto kpt2 = pkf2->mvFeatsLeft[match.trainIdx];
    cv::Mat pt1 = (cv::Mat_<float>(3, 1) << kpt1.pt.x, kpt1.pt.y, 1);
    cv::Mat pt2 = (cv::Mat_<float>(3, 1) << kpt2.pt.x, kpt2.pt.y, 1);
    th1 = 5.991 * Frame::getScaledFactor2(kpt1.octave);
    float dis1 = point2LineDistance(pt2.t() * F21, pt1);
    if (dis1 > th1)
      continue;
    th2 = 5.991 * Frame::getScaledFactor2(kpt2.octave);
    float dis2 = point2LineDistance(pt1.t() * F12, pt2);
    if (dis2 > th2)
      continue;
    goodMatches.push_back(match);
  }
  std::swap(goodMatches, matches);
  return goodMatches.size();
}

/**
 * @brief 计算点到直线的距离
 *
 * @param param 输入的参数[a, b, c]的形式（1 * 3）
 * @param point 输入的点[x, y, 1]的形式（3 * 1）
 * @return float 输出的距离
 */
float ORBMatcher::point2LineDistance(const cv::Mat &param, const cv::Mat &point)
{
  float a = param.at<float>(0, 0);
  float b = param.at<float>(0, 1);
  assert(a != 0 && b != 0);
  return std::abs(param.t().dot(point)) / std::sqrt(a * a + b * b);
}

/**
 * @brief 计算关键点到直线的距离
 *
 * @param param 输入的参数信息
 * @param point 输入的关键点信息
 * @return float 输出的距离信息
 */
float ORBMatcher::point2LineDistance(const cv::Mat &param, const cv::KeyPoint &point)
{
  cv::Mat pt = (cv::Mat_<float>(3, 1) << point.pt.x, point.pt.y, 1);
  return point2LineDistance(param, pt);
}

/**
 * @brief 设置成功匹配的地图点
 *
 * @param toMatchMps    待匹配帧的地图点
 * @param matchMps      参与匹配的地图点
 * @param matches       匹配成功后对应的id
 */
void ORBMatcher::setMapPoints(MapPoints &toMatchMps, MapPoints &matchMps, const Matches &matches)
{
  for (const auto &dmatch : matches)
  {
    auto &matchPMp = matchMps[dmatch.trainIdx];
    if (matchPMp && !matchPMp->isBad())
    {
      matchPMp->addMatchInTrack();
      toMatchMps[dmatch.queryIdx] = matchPMp;
    }
    else
    {
      matchPMp = nullptr;
    }
  }
}

/**
 * @brief 使用SAD和亚像素梯度进行精确匹配
 *
 * @param leftImage     左图（带有边界的金字塔图）
 * @param rightImage    右图（带有边界的金字塔图）
 * @param lKp           左图的特征点
 * @param rKp           右图的特征点（粗匹配成功）
 * @return float        得到的亚像素差值（精匹配失败返回0）
 */
float ORBMatcher::pixelSADMatch(const cv::Mat &leftImage, const cv::Mat &rightImage, const cv::KeyPoint &lKp, const cv::KeyPoint &rKp)
{
  std::vector<float> scores;
  float minScore = std::numeric_limits<float>::max();
  int bestL = 0;
  cv::Mat lI, rI;
  bool leftRet = getPitch(lI, leftImage, lKp, 0);
  if (!leftRet)
    return 0;
  for (int l = -mnL; l < mnL + 1; ++l)
  {
    bool rightRet = getPitch(rI, rightImage, rKp, l);
    if (!rightRet)
      continue;
    float score = SAD(lI, rI);
    if (score < minScore)
    {
      minScore = score;
      bestL = l;
    }
    scores.push_back(score);
  }
  float deltaU = 0;
  bestL += mnL;
  if (bestL > 0 && bestL < scores.size() - 1)
  {
    const float &score1 = scores[bestL - 1];
    const float &score2 = scores[bestL];
    const float &score3 = scores[bestL + 1];
    deltaU = 0.5 * (score1 - score3) / (score1 + score3 - 2 * score2);
    if (deltaU < 1 && deltaU > -1)
    {
      deltaU *= ORBExtractor::getScaledFactors()[rKp.octave];
    }
    else
    {
      deltaU = 0;
    }
  }
  return deltaU;
}

/**
 * @brief 根据输入的图像块，进行SAD的计算
 * @details
 *      1. 传入的图像的尺寸符合要求
 *      2. 减去图像块中心的点灰度来进行图像块初始化
 *      3. 计算得到的SAD越小，代表图像块之间越相似
 * @param image1 图像块1
 * @param image2 图像块2
 * @return float SAD值，越小代表越相似
 */
float ORBMatcher::SAD(const cv::Mat &image1, const cv::Mat &image2)
{
  assert(image1.rows == 2 * mnW + 1 && image2.rows == 2 * mnW + 1 && image1.cols == 2 * mnW + 1 && image2.cols == 2 * mnW + 1 && "图像不符合图像块大小的要求");
  cv::Mat i1, i2;
  image1.copyTo(i1);
  image2.copyTo(i2);
  i1.convertTo(i1, CV_32F);
  i2.convertTo(i2, CV_32F);
  cv::Mat one = cv::Mat::ones(2 * mnW + 1, 2 * mnW + 1, CV_32F);
  i1 = i1 - one * i1.at<float>(mnW, mnW);
  i2 = i2 - one * i2.at<float>(mnW, mnW);
  return cv::norm(i1, i2, cv::NORM_L1);
}

/**
 * @brief 构建行索引数据库
 * @details
 *      1. 以行为索引，将符合范围要求的右图特征点索引都放在一块
 *      2. 这里以固定的2px和金字塔缩放系数的乘积作为范围要求
 * @param pFrame 输入的帧
 * @return ORBMatcher::RowIdxDB vector<vector<size_t>>，行为索引，元素为符合行范围要求的右图特征点索引
 */
ORBMatcher::RowIdxDB ORBMatcher::createRowIndexDB(Frame *pFrame)
{
  int rows = pFrame->getLeftImage().rows;
  int cols = pFrame->getLeftImage().cols;
  RowIdxDB rowIdxDB(rows, std::vector<std::size_t>());
  const auto &rightKps = pFrame->getRightKeyPoints();
  for (std::size_t idx = 0; idx < rightKps.size(); ++idx)
  {
    const auto &kp = rightKps[idx];
    float r = 2.0 * ORBExtractor::getScaledFactors()[kp.octave];
    unsigned row = cvRound(kp.pt.y);
    unsigned maxRow = std::min(rows, cvRound(row + r + 1));
    unsigned minRow = std::max(0, cvRound(row - r));
    for (unsigned row = minRow; row < maxRow; ++row)
      rowIdxDB[row].push_back(idx);
  }
  return rowIdxDB;
}

/**
 * @brief 计算描述子之间的距离（工具）
 * 斯坦福大学的二进制统计公式
 * @param a 描述子a
 * @param b 描述子b
 * @return int 描述子之间距离
 */
int ORBMatcher::descDistance(const cv::Mat &a, const cv::Mat &b)
{
  assert(a.rows == 1 && b.rows == 1 && "两描述子的行数不为1");
  assert(a.cols == 32 && b.cols == 32 && "两描述子的列数不为32");
  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();
  int dist = 0;
  for (int i = 0; i < 8; ++i, ++pa, ++pb)
  {
    unsigned v = *pa ^ *pb;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }
  return dist;
}

/**
 * @brief 从候选描述子中，找到描述子距离最短的那个
 *
 * @param desc          被匹配的单个描述子
 * @param candidateDesc 匹配图像的所有描述子
 * @param candidateIdx  匹配图像的候选描述子索引
 * @param ratio         最优比次最优的比例
 * @return ORBMatcher::BestMatchDesc pair<size_t, int>分别代表最短距离的候选索引和最短距离
 */
ORBMatcher::BestMatchDesc ORBMatcher::getBestMatch(const cv::Mat &desc, const std::vector<cv::Mat> &candidateDesc, const std::vector<size_t> &candidateIdx,
                                                   float &ratio)
{
  assert(!candidateIdx.empty() && "候选描述子索引为空");
  int minDistance = INT_MAX;
  int secondDistance = INT_MAX;
  std::size_t minIdx = 0;
  for (const std::size_t &idx : candidateIdx)
  {
    cv::Mat cDesc = candidateDesc.at(idx);
    int distance = ORBMatcher::descDistance(desc, cDesc);
    if (distance < minDistance)
    {
      minDistance = distance;
      minIdx = idx;
    }
    else if (distance < secondDistance)
    {
      secondDistance = distance;
    }
  }
  ratio = (float)minDistance / (float)secondDistance;
  return std::make_pair(minIdx, minDistance);
}

/**
 * @brief 获取图像块
 *
 * @param pitch 输出的图像块信息
 * @param pyImg 输入的金字塔图像（带边界）
 * @param kp    输入的特征点，提供位置
 * @param L     要在x方向上偏移的像素距离
 * @return true     图像块的中心没有超过图像边界的条件
 * @return false    图像块的中心超过了图像边界的条件
 */
bool ORBMatcher::getPitch(cv::Mat &pitch, const cv::Mat &pyImg, const cv::KeyPoint &kp, int L)
{
  const auto &scaleFactors = ORBExtractor::getScaledFactors();
  int x = cvFloor(kp.pt.x / scaleFactors[kp.octave]) + L;
  int y = cvFloor(kp.pt.y / scaleFactors[kp.octave]);

  pitch = pyImg.rowRange(y - mnW, y + mnW + 1);
  pitch = pitch.colRange(x - mnW, x + mnW + 1);
  return true;
}

void ORBMatcher::verifyAngle(std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keyPoints1, const std::vector<cv::KeyPoint> &keyPoints2)
{
  HistBin hist(mnBinNum);
  for (const auto &dmatch : matches)
  {
    float diff = keyPoints1[dmatch.queryIdx].angle - keyPoints2[dmatch.trainIdx].angle;
    diff = diff >= 0 ? diff : 360 + diff;
    int bin = diff / (360 / mnBinNum);
    if (bin == 30)
      bin = 0;
    hist[bin].push_back(dmatch);
  }
  std::set<std::size_t> goodBinIds;
  for (std::size_t idx = 0; idx < mnBinChoose; ++idx)
  {
    int maxSize = 0;
    std::size_t maxId = 0;
    bool bInit = false;
    for (std::size_t id = 0; id < mnBinNum; ++id)
    {
      int binSize = hist[id].size();
      if (goodBinIds.find(id) != goodBinIds.end())
        continue;
      if (binSize > maxSize)
      {
        maxId = id;
        maxSize = binSize;
        bInit = true;
      }
    }
    if (bInit)
      /// 代表所有的bin的大小都为0，添加与否都没意义了
      goodBinIds.insert(maxId);
  }
  std::vector<cv::DMatch> ret;
  for (auto &idx : goodBinIds)
    std::copy(hist[idx].begin(), hist[idx].end(), std::back_inserter(ret));
  std::swap(ret, matches);
}

/**
 * @brief 用于展示匹配结果
 *
 * @param image1    输入的图像1
 * @param image2    输入的图像2
 * @param keypoint1 输入的图像1的关键点
 * @param keypoint2 输入的图像2的关键点
 * @param matches   输入的两图像之间的匹配信息
 */
void ORBMatcher::showMatches(const cv::Mat &image1, const cv::Mat &image2, const std::vector<cv::KeyPoint> &keypoint1,
                             const std::vector<cv::KeyPoint> &keypoint2, const std::vector<cv::DMatch> &matches)
{
  cv::Mat showImage;
  std::vector<cv::Mat> imgs{image1, image2};
  cv::hconcat(imgs, showImage);
  cv::cvtColor(showImage, showImage, cv::COLOR_GRAY2BGR);
  std::vector<cv::KeyPoint> rightKps, leftKps;
  for (const auto &dmatch : matches)
  {
    const auto &lkp = keypoint1[dmatch.queryIdx];
    cv::KeyPoint rkp = keypoint2[dmatch.trainIdx];
    rkp.pt.x = rkp.pt.x + image1.cols;
    cv::line(showImage, lkp.pt, rkp.pt, cv::Scalar(255, 0, 0));
    rightKps.push_back(rkp);
    leftKps.push_back(lkp);
  }
  cv::drawKeypoints(showImage, leftKps, showImage, cv::Scalar(0, 255, 0));
  cv::drawKeypoints(showImage, rightKps, showImage, cv::Scalar(0, 0, 255));
  cv::imshow("showImage", showImage);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

int ORBMatcher::mnMaxThreshold = 100;
int ORBMatcher::mnMinThreshold = 50;
int ORBMatcher::mnMeanThreshold = 75;
int ORBMatcher::mnW = 5;
int ORBMatcher::mnL = 5;
int ORBMatcher::mnBinNum = 30;
int ORBMatcher::mnBinChoose = 3;
int ORBMatcher::mnFarParam = 35;

} // namespace ORB_SLAM2_ROS2