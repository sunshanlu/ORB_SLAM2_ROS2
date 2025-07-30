#include <cmath>
#include <unordered_set>

#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <opencv2/core/eigen.hpp>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/Optimizer.h"
#include "ORB_SLAM2/Sim3Solver.h"

namespace ORB_SLAM2_ROS2
{

/**
 * @brief 跟踪线程部分的优化器，仅优化帧位姿
 * @details
 *      1. 使用单目误差边和双目误差边两种方式进行优化
 *      2. 优化过程中，设置缩放因子倒数的平方作为信息矩阵（金字塔层级越高，比重越小）
 *      3. 在优化过程之前，需要设置帧的位姿
 *      4. 优化过程中会将明显的外点进行剔除（误差较大的边对应的点 + 投影超出范围的地图点）
 * @param pFrame    输入的待优化位姿的帧
 * @return int      输出优化内点的个数
 */
int Optimizer::OptimizePoseOnly(Frame::SharedPtr pFrame)
{
  auto lm = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BSSE3>(std::make_unique<LSSE3Dense>()));

  g2o::SparseOptimizer graph;
  graph.setAlgorithm(lm);

  auto poseVertex = new g2o::VertexSE3Expmap();
  poseVertex->setId(0);
  graph.addVertex(poseVertex);

  std::map<std::size_t, g2o::EdgeSE3ProjectXYZOnlyPose *> monoEdges;
  std::map<std::size_t, g2o::EdgeStereoSE3ProjectXYZOnlyPose *> stereoEdges;
  std::vector<bool> inLier(pFrame->mvFeatsLeft.size(), true);

  /// 添加位姿节点和误差边
  int edges = 0;
  std::vector<cv::Mat> mapPointPoses;
  auto mapPoints = pFrame->getMapPoints();
  auto &kps = pFrame->getLeftKeyPoints();
  std::vector<double> monoEdgeErrors;
  std::vector<double> stereoEdgeErrors;
  std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> monoEdgesVec;
  std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> stereoEdgesVec;

  for (std::size_t idx = 0; idx < mapPoints.size(); ++idx)
  {
    auto &pMp = mapPoints[idx];
    const auto &rightU = pFrame->getRightU(idx);
    const auto &fKp = kps[idx];
    cv::Mat pos;
    if (pMp && !pMp->isBad())
    {
      pos = pMp->getPos();
      auto rk = new g2o::RobustKernelHuber();
      if (rightU < 0)
      {
        rk->setDelta(deltaMono);
        auto edgeMono = new g2o::EdgeSE3ProjectXYZOnlyPose();
        edgeMono->setVertex(0, poseVertex);
        edgeMono->fx = Camera::mfFx;
        edgeMono->fy = Camera::mfFy;
        edgeMono->cx = Camera::mfCx;
        edgeMono->cy = Camera::mfCy;
        edgeMono->Xw << (double)pos.at<float>(0), (double)pos.at<float>(1), (double)pos.at<float>(2);
        edgeMono->setMeasurement(g2o::Vector2((double)fKp.pt.x, (double)fKp.pt.y));
        edgeMono->setInformation(Eigen::Matrix2d::Identity() * pFrame->getScaledFactorInv2(fKp.octave));
        edgeMono->setRobustKernel(rk);
        monoEdges.insert(std::make_pair(idx, edgeMono));
        graph.addEdge(edgeMono);
        edgeMono->computeError();
        double error = edgeMono->chi2();
        monoEdgeErrors.push_back(error);
        monoEdgesVec.push_back(edgeMono);
      }
      else
      {
        rk->setDelta(deltaStereo);
        auto edgeStereo = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
        edgeStereo->setVertex(0, poseVertex);
        edgeStereo->fx = Camera::mfFx;
        edgeStereo->fy = Camera::mfFy;
        edgeStereo->cx = Camera::mfCx;
        edgeStereo->cy = Camera::mfCy;
        edgeStereo->bf = Camera::mfBf;
        edgeStereo->Xw << (double)pos.at<float>(0), (double)pos.at<float>(1), (double)pos.at<float>(2);
        edgeStereo->setMeasurement(g2o::Vector3((double)fKp.pt.x, (double)fKp.pt.y, (double)rightU));
        edgeStereo->setInformation(Eigen::Matrix3d::Identity() * pFrame->getScaledFactorInv2(fKp.octave));
        edgeStereo->setRobustKernel(rk);
        stereoEdges.insert(std::make_pair(idx, edgeStereo));
        graph.addEdge(edgeStereo);
        edgeStereo->computeError();
        double error = edgeStereo->chi2();
        stereoEdgeErrors.push_back(error);
        stereoEdgesVec.push_back(edgeStereo);
      }
      ++edges;
    }
    else
    {
      pMp = nullptr;
      inLier[idx] = false;
    }
    mapPointPoses.push_back(pos);
  }

  auto se3 = Converter::ConvertTcw2SE3(pFrame->mRcw, pFrame->mtcw);
  int nBad = 0;

  for (int i = 0; i < 4; ++i)
  {
    nBad = 0;
    poseVertex->setEstimate(se3);
    graph.initializeOptimization(0);
    graph.optimize(10);

    /// 寻找超出要求的误差边
    for (auto &item : monoEdges)
    {
      auto &edge = item.second;
      const auto &octave = pFrame->mvFeatsLeft[item.first].octave;
      float sigma2 = pFrame->getScaledFactor2(octave);
      if (!inLier[item.first])
      {
        edge->computeError();
      }
      auto error = edge->chi2();
      if (edge->chi2() > 5.991 * sigma2)
      {
        inLier[item.first] = false;
        edge->setLevel(1);
        ++nBad;
      }
      else
      {
        inLier[item.first] = true;
        edge->setLevel(0);
      }
      if (i == 2)
        edge->setRobustKernel(nullptr);
    }
    for (auto &item : stereoEdges)
    {
      auto &edge = item.second;
      const auto &octave = pFrame->mvFeatsLeft[item.first].octave;
      float sigma2 = pFrame->getScaledFactor2(octave);
      if (!inLier[item.first])
      {
        edge->computeError();
      }
      auto error = edge->chi2();
      if (edge->chi2() > 7.815 * sigma2)
      {
        inLier[item.first] = false;
        edge->setLevel(1);
        ++nBad;
      }
      else
      {
        inLier[item.first] = true;
        edge->setLevel(0);
      }
      if (i == 2)
        edge->setRobustKernel(nullptr);
    }
  }
  for (std::size_t idx = 0; idx < inLier.size(); ++idx)
  {
    if (!inLier[idx])
      continue;
    bool isPositive = false;
    auto pointUV = pFrame->project2UV(mapPointPoses[idx], isPositive);
    if (!isPositive || pointUV.x > pFrame->mfMaxU || pointUV.x < 0 || pointUV.y > pFrame->mfMaxV || pointUV.y < 0)
    {
      inLier[idx] = false;
      ++nBad;
    }
  }
  for (std::size_t idx = 0; idx < inLier.size(); ++idx)
  {
    if (!inLier[idx])
      pFrame->mvpMapPoints[idx] = nullptr;
    else
    {
      pFrame->mvpMapPoints[idx]->addInlierInTrack();
    }
  }
  auto se3Optimized = poseVertex->estimate();
  pFrame->setPose(Converter::ConvertSE32Tcw(se3Optimized));
  return edges - nBad;
}

/**
 * @brief 局部建图线程中的局部地图优化
 * @details
 *      1. 找到当前关键帧的一阶相连关键帧
 *      2. 使用地图点的方式找到二阶相连关键帧
 *      3. 将二阶相连关键帧对应的顶点固定（也包括id为0的关键帧进行固定）
 *      4. 进行两次优化
 *          (1) 第一次优化5次
 *          (2) 第二次优化10次
 *          (3) 在每次优化后，将误差边较大的部分不参与下次优化
 *      5. 对误差较大的边对应的地图点进行观测剔除
 *          (1) 关键帧：对应的地图点位置设置为nullptr
 *          (2) 地图点：对应的Observe进行删除
 *      6. 设置关键帧位姿和地图点位置
 * @param pkframe   输入的局部建图线程的当前关键帧
 * @param isStop    是否终止BA（当跟踪线程插入关键帧的时候，为true）
 *      1. g2o会接受一个bool类型的指针，在每一次迭代开始前进行判断是否要开始优化
 *      2. 在第一阶段的优化前，判断是否需要停止BA，如果停止，直接return
 *      3. 在第二阶段的优化前，判断是否需要停止BA，如果停止，直接跳过第二次优化，直接进行下一步操作
 */
void Optimizer::OptimizeLocalMap(KeyFramePtr pkframe, bool &isStop)
{
  g2o::SparseOptimizer optimizer;
  auto lm = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BSSE3>(std::make_unique<LSSE3Eigen>()));
  optimizer.setAlgorithm(lm);
  optimizer.setForceStopFlag(&isStop);

  auto vNoFixedGroup = pkframe->getConnectedKfs(0);
  vNoFixedGroup.push_back(pkframe);

  std::map<MapPointPtr, g2o::VertexPointXYZ *> mLandMarks;
  std::map<KeyFramePtr, g2o::VertexSE3Expmap *> mNoFixedFrames;
  std::map<KeyFramePtr, g2o::VertexSE3Expmap *> mFixedFrames;
  int nFrames = KeyFrame::getMaxID() + 1;
  int edgeID = 0;
  EdgeDB edgeDB;

  // 建立非固定顶点数据库和地图点顶点数据库
  std::set<MapPointPtr> sGroupMps;
  for (auto &pNoFixed : vNoFixedGroup)
  {
    cv::Mat Rcw, tcw;
    pNoFixed->getPose(Rcw, tcw);
    auto vFrame = new g2o::VertexSE3Expmap();
    vFrame->setId(pNoFixed->getID());
    vFrame->setEstimate(Converter::ConvertTcw2SE3(Rcw, tcw));
    vFrame->setFixed(pNoFixed->getID() == 0);
    optimizer.addVertex(vFrame);
    mNoFixedFrames.insert({pNoFixed, vFrame});
    auto vpMps = pNoFixed->getMapPoints();
    for (auto &pMp : vpMps)
    {
      if (pMp && !pMp->isBad())
        sGroupMps.insert(pMp);
    }
  }
  for (auto &pMp : sGroupMps)
  {
    auto vPoint = new g2o::VertexPointXYZ();
    vPoint->setId(pMp->getID() + nFrames);
    vPoint->setMarginalized(true);
    vPoint->setEstimate(Converter::ConvertPw2Vector3(pMp->getPos()));
    optimizer.addVertex(vPoint);
    mLandMarks.insert({pMp, vPoint});

    auto pMpObs = pMp->getObservation();
    for (auto &obs : pMpObs)
    {
      g2o::VertexSE3Expmap *vFrame;
      auto pkf = obs.first.lock();
      if (!pkf || pkf->isBad())
        continue;

      if (mNoFixedFrames.find(pkf) != mNoFixedFrames.end())
        vFrame = mNoFixedFrames[pkf];
      else if (mFixedFrames.find(pkf) != mFixedFrames.end())
        vFrame = mFixedFrames[pkf];
      else
      {
        cv::Mat Rcw, tcw;
        pkf->getPose(Rcw, tcw);
        vFrame = new g2o::VertexSE3Expmap();
        vFrame->setId(pkf->getID());
        vFrame->setEstimate(Converter::ConvertTcw2SE3(Rcw, tcw));
        vFrame->setFixed(true);
        optimizer.addVertex(vFrame);
        mFixedFrames.insert({pkf, vFrame});
      }
      double rightU = pkf->getRightU(obs.second);
      auto &kp = pkf->getLeftKeyPoint(obs.second);
      auto rk = new g2o::RobustKernelHuber();
      if (rightU > 0)
      {
        rk->setDelta(deltaStereo);
        auto eStereo = new g2o::EdgeStereoSE3ProjectXYZ();
        eStereo->setId(edgeID++);
        eStereo->setInformation(Eigen::Matrix3d::Identity() * pkf->getScaledFactorInv2(kp.octave));
        eStereo->setVertex(0, vPoint);
        eStereo->setVertex(1, vFrame);
        eStereo->fx = Camera::mfFx;
        eStereo->fy = Camera::mfFy;
        eStereo->cx = Camera::mfCx;
        eStereo->cy = Camera::mfCy;
        eStereo->bf = Camera::mfBf;
        eStereo->setMeasurement(Eigen::Vector3d(kp.pt.x, kp.pt.y, rightU));
        eStereo->setRobustKernel(rk);
        optimizer.addEdge(eStereo);
        edgeDB.insert({eStereo, std::make_tuple(pMp, pkf, obs.second, false)});
      }
      else
      {
        rk->setDelta(deltaMono);
        auto eMono = new g2o::EdgeSE3ProjectXYZ();
        eMono->setId(edgeID++);
        eMono->setInformation(Eigen::Matrix2d::Identity() * pkf->getScaledFactorInv(kp.octave));
        eMono->setVertex(0, vPoint);
        eMono->setVertex(1, vFrame);
        eMono->fx = Camera::mfFx;
        eMono->fy = Camera::mfFy;
        eMono->cx = Camera::mfCx;
        eMono->cy = Camera::mfCy;
        eMono->setMeasurement(Eigen::Vector2d(kp.pt.x, kp.pt.y));
        eMono->setRobustKernel(rk);
        optimizer.addEdge(eMono);
        edgeDB.insert({eMono, std::make_tuple(pMp, pkf, obs.second, true)});
      }
    }
  }
  if (isStop)
    return;

  optimizer.initializeOptimization(0);
  optimizer.optimize(5);
  if (!isStop)
  {
    for (auto &item : edgeDB)
    {
      auto pkf = std::get<1>(item.second);
      auto edge = item.first;
      bool isMono = std::get<3>(item.second);
      if (isMono)
      {
        auto edge = dynamic_cast<g2o::EdgeSE3ProjectXYZ *>(item.first);
        if (edge->chi2() > 5.991 || !edge->isDepthPositive())
          edge->setLevel(1);
        edge->setRobustKernel(nullptr);
      }
      else
      {
        auto edge = dynamic_cast<g2o::EdgeStereoSE3ProjectXYZ *>(item.first);
        if (edge->chi2() > 7.815 || !edge->isDepthPositive())
          edge->setLevel(1);
        edge->setRobustKernel(nullptr);
      }
    }
    optimizer.initializeOptimization();
    optimizer.optimize(10);
  }

  std::map<KeyFramePtr, std::vector<std::pair<MapPointPtr, std::size_t>>> vToProcess;
  for (auto &item : edgeDB)
  {
    auto &pMp = std::get<0>(item.second);
    auto &pkf = std::get<1>(item.second);
    auto &idx = std::get<2>(item.second);
    bool isMono = std::get<3>(item.second);
    if (isMono)
    {
      auto edge = dynamic_cast<g2o::EdgeSE3ProjectXYZ *>(item.first);
      edge->computeError();
      if (edge->chi2() > 5.991 || !edge->isDepthPositive())
      {
        vToProcess[pkf].push_back(std::make_pair(pMp, idx));
      }
    }
    else
    {
      auto edge = dynamic_cast<g2o::EdgeStereoSE3ProjectXYZ *>(item.first);
      edge->computeError();
      if (edge->chi2() > 7.815 || !edge->isDepthPositive())
      {
        vToProcess[pkf].push_back(std::make_pair(pMp, idx));
      }
    }
  }

  int nBad = 0;
  bool bSetAndErase = true;
  for (auto &item : vToProcess)
  {
    int nGoodMp = 0;
    auto vMps = item.first->getMapPoints();
    for (auto &pMp : vMps)
    {
      if (pMp && !pMp->isBad())
        ++nGoodMp;
    }
    if (item.second.size() / (float)nGoodMp > 0.3)
      ++nBad;
  }
  if (nBad / (vToProcess.size() + 1e-5) > 0.2)
    bSetAndErase = false;

  if (bSetAndErase)
  {
    for (auto &item : vToProcess)
    {
      auto pkf = item.first;
      for (auto &era : item.second)
      {
        pkf->setMapPoint(era.second, nullptr);
        era.first->eraseObservetion(pkf);
      }
    }

    for (auto &frame : mNoFixedFrames)
    {
      auto &pkf = frame.first;
      auto &vertex = frame.second;
      cv::Mat Tcw = Converter::ConvertSE32Tcw(vertex->estimate());
      if (pkf && !pkf->isBad())
        pkf->setPose(Tcw);
    }
    for (auto &landmark : mLandMarks)
    {
      auto &pMp = landmark.first;
      auto &vertex = landmark.second;
      cv::Mat pos = Converter::ConvertVector32Pw(vertex->estimate());
      if (pMp && !pMp->isBad() && pMp->isInMap())
      {
        pMp->setPos(pos);
        pMp->updateDescriptor();
        pMp->updateNormalAndDepth();
      }
    }
    KeyFrame::updateConnections(pkframe);
  }
}

/**
 * @brief 优化SIM3相似变换
 * @details
 *      1. 进行优化的前提是，需要使用SIM3的优化器获取初步的SIM3结果
 *      2. 构建g2o的SIM3优化器，注意BlockSolver和LinearSolver都发生了改变
 *      3. 构建SIM3顶点，注意这里的SIM3代表的是Scm
 *      4. 构建3d顶点和对应的边
 *          1) 3d顶点指的是两相机坐标系下的3d位置
 *          2) 正向投影边：m坐标系向c像素坐标系投影，Scm保持原状即可
 *          3) 反向投影边：c坐标系向m像素坐标系投影，Scm需要inverse
 *      5. 首先，进行5次优化，然后将误差大的边，进行清除（remove，不是改变level）
 *      6. 如果有被清除的边产生，则认为优化的结果不好，进行10次优化，否则进行5次
 *      7. 统计优化后内点的数目
 * @param pCurr         输入的回环闭合线程的当前关键帧
 * @param pMatch        输入的回环闭合线程的回环闭合关键帧
 * @param inLier        输入的SIM3优化器认为是内点和sim3投影获取的匹配
 * @param g2oScm        输入输出的SIM3变换
 * @param bFixedScale   是否固定尺度，如果固定尺度，则固定为1
 * @return int  输出优化SIM3的内点数目
 */
int Optimizer::OptimizeSim3(KeyFramePtr pCurr, KeyFramePtr pMatch, Matches &inLier, Sim3Ret &g2oScm, bool bFixedScale)
{
  auto lm = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BSSIM3>(std::make_unique<LSSim3Dense>()));

  g2o::SparseOptimizer graph;
  graph.setAlgorithm(lm);

  int vID = 0, eID = 0;
  g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
  vSim3->setId(vID++);
  vSim3->setEstimate(Converter::ConvertSim3G2o(g2oScm));
  vSim3->_fix_scale = bFixedScale;
  vSim3->_principle_point1[0] = Camera::mfCx;
  vSim3->_principle_point1[1] = Camera::mfCy;
  vSim3->_principle_point2[0] = Camera::mfCx;
  vSim3->_principle_point2[1] = Camera::mfCy;
  vSim3->_focal_length1[0] = Camera::mfFx;
  vSim3->_focal_length1[1] = Camera::mfFy;
  vSim3->_focal_length2[0] = Camera::mfFx;
  vSim3->_focal_length2[1] = Camera::mfFy;
  graph.addVertex(vSim3);

  cv::Mat Rcw, tcw, Rmw, tmw;
  pCurr->getPose(Rcw, tcw);
  pMatch->getPose(Rmw, tmw);
  std::vector<g2o::EdgeSim3ProjectXYZ *> vpEdgesMC;
  std::vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdgesCM;
  std::vector<bool> vbIsInlier(inLier.size(), true);
  for (std::size_t idx = 0; idx < inLier.size(); ++idx)
  {
    const auto &match = inLier[idx];
    const int &cID = match.queryIdx;
    const int &mID = match.trainIdx;
    MapPoint::SharedPtr cP = pCurr->getMapPoint(cID);
    MapPoint::SharedPtr mP = pMatch->getMapPoint(mID);
    if (!cP || cP->isBad() || !cP->isInMap())
    {
      vbIsInlier[idx] = false;
      continue;
    }
    if (!mP || mP->isBad() || !cP->isInMap())
    {
      vbIsInlier[idx] = false;
      continue;
    }
    cv::Mat cPw = cP->getPos();
    cv::Mat mPw = mP->getPos();
    cv::Mat cPc = Rcw * cPw + tcw;
    cv::Mat mPc = Rmw * mPw + tmw;

    auto vPointC = new g2o::VertexPointXYZ();
    vPointC->setId(vID++);
    vPointC->setFixed(true);
    vPointC->setEstimate(Converter::ConvertPw2Vector3(cPc));
    graph.addVertex(vPointC);

    auto vPointM = new g2o::VertexPointXYZ();
    vPointM->setId(vID++);
    vPointM->setFixed(true);
    vPointM->setEstimate(Converter::ConvertPw2Vector3(mPc));
    graph.addVertex(vPointM);

    const auto &ckp = pCurr->getLeftKeyPoint(cID);
    const auto &mkp = pMatch->getLeftKeyPoint(mID);

    /// 正向投影，回环闭合关键帧投影到当前关键帧
    auto e1 = new g2o::EdgeSim3ProjectXYZ();
    auto rk1 = new g2o::RobustKernelHuber();
    rk1->setDelta(deltaSim3);
    e1->setVertex(0, vPointM);
    e1->setVertex(1, vSim3);
    e1->setMeasurement(g2o::Vector2((double)ckp.pt.x, (double)ckp.pt.y));
    e1->setInformation(Eigen::Matrix2d::Identity() * pCurr->getScaledFactorInv2(ckp.octave));
    e1->setRobustKernel(rk1);

    /// 反向投影，当前关键帧投影到回环闭合关键帧
    auto e2 = new g2o::EdgeInverseSim3ProjectXYZ();
    auto rk2 = new g2o::RobustKernelHuber();
    rk2->setDelta(deltaSim3);
    e2->setVertex(0, vPointC);
    e2->setVertex(1, vSim3);
    e2->setMeasurement(g2o::Vector2((double)mkp.pt.x, (double)mkp.pt.y));
    e2->setInformation(Eigen::Matrix2d::Identity() * pCurr->getScaledFactorInv2(mkp.octave));
    e2->setRobustKernel(rk2);

    graph.addEdge(e1);
    graph.addEdge(e2);
    vpEdgesMC.push_back(e1);
    vpEdgesCM.push_back(e2);
  }

  std::vector<cv::DMatch> newInlier;
  for (std::size_t idx = 0; idx < vbIsInlier.size(); ++idx)
  {
    if (vbIsInlier[idx])
    {
      newInlier.push_back(inLier[idx]);
    }
  }
  std::vector<bool> vbNewIsInlier(newInlier.size(), true);
  graph.initializeOptimization(0);
  graph.optimize(5);

  /// 移除误差较大的边
  int nBad = 0;
  int nEdges = vpEdgesCM.size();
  for (int i = 0; i < nEdges; ++i)
  {
    auto &e1 = vpEdgesMC[i];
    auto &e2 = vpEdgesCM[i];
    if (e1->chi2() > 9.210 || e2->chi2() > 9.210)
    {
      graph.removeEdge(e1);
      graph.removeEdge(e2);
      e1 = nullptr;
      e2 = nullptr;
      ++nBad;
    }
  }

  if (nEdges - nBad < 20)
    return 0;
  int iteration = nBad ? 10 : 5;
  graph.initializeOptimization();
  graph.optimize(iteration);
  int nInliers = 0;
  for (int i = 0; i < nEdges; ++i)
  {
    const auto &e1 = vpEdgesMC[i];
    const auto &e2 = vpEdgesCM[i];
    if (!e1 || !e2)
    {
      vbNewIsInlier[i] = false;
      continue;
    }
    if (e1->chi2() <= 9.210 && e2->chi2() <= 9.210)
    {
      ++nInliers;
    }
    else
    {
      vbNewIsInlier[i] = false;
    }
  }
  std::vector<cv::DMatch> vNewNewInlier;
  for (std::size_t idx = 0; idx < newInlier.size(); ++idx)
  {
    if (vbNewIsInlier[idx])
    {
      vNewNewInlier.push_back(newInlier[idx]);
    }
  }
  std::swap(vNewNewInlier, inLier);
  Converter::ConvertG2o2Sim3(vSim3->estimate()).copyTo(g2oScm);
  return nInliers;
}

/**
 * @brief 将opencv的Rcw和tcw转换为g2o的SE3
 *
 * @param RcwCV 输入的Rcw
 * @param tcwCV 输入的tcw
 * @return g2o::SE3Quat 输出的SE3
 */
g2o::SE3Quat Converter::ConvertTcw2SE3(const cv::Mat &RcwCV, const cv::Mat &tcwCV)
{
  Eigen::Matrix3d RcwEigen;
  Eigen::Vector3d tcwEigen;
  RcwEigen << (double)RcwCV.at<float>(0, 0), (double)RcwCV.at<float>(0, 1), (double)RcwCV.at<float>(0, 2), (double)RcwCV.at<float>(1, 0),
      (double)RcwCV.at<float>(1, 1), (double)RcwCV.at<float>(1, 2), (double)RcwCV.at<float>(2, 0), (double)RcwCV.at<float>(2, 1), (double)RcwCV.at<float>(2, 2);
  tcwEigen.x() = (double)tcwCV.at<float>(0, 0);
  tcwEigen.y() = (double)tcwCV.at<float>(1, 0);
  tcwEigen.z() = (double)tcwCV.at<float>(2, 0);
  Eigen::Quaterniond qcwEigen(RcwEigen);
  qcwEigen.normalize();
  return g2o::SE3Quat(qcwEigen, tcwEigen);
}

/**
 * @brief 将g2o类型的位姿矩阵转换为OpenCV类型表示的位姿矩阵
 *
 * @param SE3   输入的g2o类型的位姿矩阵
 * @return cv::Mat 输出的OpenCV类型的位姿矩阵
 */
cv::Mat Converter::ConvertSE32Tcw(const g2o::SE3Quat &SE3)
{
  cv::Mat Tcw(4, 4, CV_32F);
  Tcw.at<float>(0, 0) = (float)SE3.rotation().matrix()(0, 0);
  Tcw.at<float>(0, 1) = (float)SE3.rotation().matrix()(0, 1);
  Tcw.at<float>(0, 2) = (float)SE3.rotation().matrix()(0, 2);
  Tcw.at<float>(1, 0) = (float)SE3.rotation().matrix()(1, 0);
  Tcw.at<float>(1, 1) = (float)SE3.rotation().matrix()(1, 1);
  Tcw.at<float>(1, 2) = (float)SE3.rotation().matrix()(1, 2);
  Tcw.at<float>(2, 0) = (float)SE3.rotation().matrix()(2, 0);
  Tcw.at<float>(2, 1) = (float)SE3.rotation().matrix()(2, 1);
  Tcw.at<float>(2, 2) = (float)SE3.rotation().matrix()(2, 2);

  Tcw.at<float>(0, 3) = (float)SE3.translation().x();
  Tcw.at<float>(1, 3) = (float)SE3.translation().y();
  Tcw.at<float>(2, 3) = (float)SE3.translation().z();

  Tcw.at<float>(3, 0) = 0.0f;
  Tcw.at<float>(3, 1) = 0.0f;
  Tcw.at<float>(3, 2) = 0.0f;
  Tcw.at<float>(3, 3) = 1.0f;
  return Tcw;
}

/**
 * @brief 将cv::Mat类型的Pw转换成g2o::Vector3
 *
 * @param Pw 输入的cv::Mat类型的Pw
 * @return g2o::Vector3 输出的g2o::Vector3类型的Pw
 */
g2o::Vector3 Converter::ConvertPw2Vector3(const cv::Mat &Pw)
{
  float x = Pw.at<float>(0);
  float y = Pw.at<float>(1);
  float z = Pw.at<float>(2);
  return g2o::Vector3((double)x, (double)y, (double)z);
}

/**
 * @brief 将g2o::Vector3类型的Pw转换为cv::Mat类型
 *
 * @param Pw 输入的g2o::Vector3类型的Pw
 * @return cv::Mat 输出的cv::Mat类型的Pw
 */
cv::Mat Converter::ConvertVector32Pw(const g2o::Vector3 &Pw)
{
  float x = Pw[0];
  float y = Pw[1];
  float z = Pw[2];
  return (cv::Mat_<float>(3, 1) << x, y, z);
}

/**
 * @brief 计算某个数据的3/4分位数
 *
 * @param data 输入的计算分位数的数据(排序好的)
 * @return double 输出的3/4分位数的值
 */
double Optimizer::ComputeThirdQuartile(const std::vector<double> &data)
{
  std::size_t N = data.size();
  assert(N > 0);
  std::size_t targetId = std::floor((N - 1) * 0.75);
  double remainder = (N - 1) * 0.75 - targetId;
  if (remainder == 0)
  {
    return data[targetId];
  }
  else
    return data[targetId] * (1.0 - remainder) + remainder * (data[targetId + 1]);
}

/**
 * @brief 优化本质图
 * @details
 *      1. 回环闭合产生新的共视关系
 *          1) 与当前关键帧或闭合关键帧产生直接共视
 *          2) 共视关系超过graphTh
 *      2. 生成树
 *          1) 所有关键帧和其父关键帧的共视关系
 *          2) 以往产生的所有回环闭合边（用于闭环生成树）
 *      3. 共视权重超过权重graphTh的边
 *          1) 去除1中共视关系超过graphTh的边
 *          2) 去除其中的父子关键帧共视+回环闭合共视
 *      4. 优化20次后，进行地图点的矫正和关键帧位姿的确定
 *          1) 关键帧位姿确定，使用Sim3矩阵的降维
 *          2) 地图点的矫正
 *              a) 如何地图点的mpLoopCorrected存在，使用mpLoopCorrected矫正
 *              b) 否则，使用地图点的参考关键帧进行位置矫正
 *              c) 矫正方式，与correctLoop中使用的方法一致
 * @param mLoopConnections  输入的由于回环闭合产生的新连接
 * @param mpMap             输入的地图，用于获取关键帧和地图点
 * @param pLoopKf           输入的回环闭合关键帧
 * @param pCurrKf           输入的当前关键帧
 * @param graphTh           输入的本质图连接权重的最小阈值
 * @param mCorrectedG2oScw           输入的经过位姿矫正的sim3矩阵，顶点定义时使用
 * @param bFixedScale       输入的是否固定尺度标识
 */
void Optimizer::optimizeEssentialGraph(const LoopConnection &mLoopConnections, MapPtr mpMap, KeyFramePtr pLoopKf, KeyFramePtr pCurrKf, const int &graphTh,
                                       const KeyFrameAndSim3 &mCorrectedG2oScw, const KeyFrameAndSim3 &mNoCorrectedG2oScw, bool bFixScale)
{
  /// step1: 构建优化器
  auto lm = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BSSIM3>(std::make_unique<LSSim3Eigen>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(lm);
  // optimizer.setVerbose(true);

  /// step2: 添加关键帧顶点
  auto eInformation = Eigen::Matrix<double, 7, 7>::Identity();
  auto correctedEndIter = mCorrectedG2oScw.end();
  auto noCorrectedEndIter = mNoCorrectedG2oScw.end();
  std::size_t nMaxID = KeyFrame::getMaxID();
  auto vAllKeyFrames = mpMap->getAllKeyFrames();
  std::vector<KeyFramePtr> vKeyFrames(nMaxID + 1, nullptr);
  std::vector<g2o::Sim3> vScwg(nMaxID + 1); ///< 本质图优化前的sim3矩阵
  std::vector<Sim3Ret> vScw(nMaxID + 1);    ///< 本质图优化前的sim3矩阵
  std::vector<Sim3Ret> vG2oSwc(nMaxID + 1); ///< 本质图优化后的sim3矩阵
  std::vector<g2o::VertexSim3Expmap *> vpSim3Vertexes(nMaxID + 1, nullptr);
  int vNum = 0;
  for (auto &pKf : vAllKeyFrames)
  {
    if (!pKf || pKf->isBad())
      continue;
    int vID = pKf->getID();
    vKeyFrames[vID] = pKf;

    Sim3Ret Siw;
    if (mCorrectedG2oScw.find(pKf) != correctedEndIter)
    {
      Siw = mCorrectedG2oScw.at(pKf);
    }
    else
    {
      cv::Mat Rcw, tcw;
      pKf->getPose(Rcw, tcw);
      Siw = Sim3Ret(Rcw, tcw, 1.f);
    }
    g2o::Sim3 Siwg = Converter::ConvertSim3G2o(Siw);
    auto vi = new g2o::VertexSim3Expmap();
    vi->setId(vID);
    vi->_fix_scale = bFixScale;
    vi->setEstimate(Siwg);
    vScwg[vID] = Siwg;
    vScw[vID] = Siw;
    vpSim3Vertexes[vID] = vi;
    if (vID == pLoopKf->getID())
      vi->setFixed(true);
    optimizer.addVertex(vi);
    ++vNum;
  }

  /// step3: 添加观测边
  std::set<std::pair<std::size_t, std::size_t>> sAlreadyConnected;

  /// step3.1: 添加回环闭合产生新的共视关系
  int eID = 0;
  for (const auto &c : mLoopConnections)
  {
    auto pkf1 = c.first;
    auto sConnectedPkf2 = c.second;
    auto id1 = pkf1->getID();
    const g2o::Sim3 S1w = vScwg[id1];
    const g2o::Sim3 Sw1 = S1w.inverse();
    for (auto &pkf2 : sConnectedPkf2)
    {
      if (!pkf2 || pkf2->isBad())
        continue;
      int weight = pkf1->getWeight(pkf2);
      if ((pkf1 != pCurrKf || pkf2 != pLoopKf) && weight < graphTh)
        continue;
      auto id2 = pkf2->getID();
      const g2o::Sim3 S2w = vScwg[id2];
      const g2o::Sim3 S21 = S2w * Sw1;

      auto e = new g2o::EdgeSim3();
      e->setVertex(0, vpSim3Vertexes[id1]);
      e->setVertex(1, vpSim3Vertexes[id2]);
      e->setMeasurement(S21);
      e->setId(eID++);
      e->setInformation(eInformation);
      optimizer.addEdge(e);
      sAlreadyConnected.insert({id1, id2});
    }
  }

  /// step3.2: 添加生成树
  for (std::size_t id1 = 0; id1 <= nMaxID; ++id1)
  {
    auto &pkf1 = vKeyFrames[id1];
    if (!pkf1 || pkf1->isBad())
      continue;

    auto pEdgesKFs = pkf1->getLoopEdges();
    g2o::Sim3 Sw1;
    if (mNoCorrectedG2oScw.find(pkf1) != noCorrectedEndIter)
      Sw1 = Converter::ConvertSim3G2o(mNoCorrectedG2oScw.at(pkf1)).inverse();
    else
      Sw1 = vScwg[id1].inverse();

    auto parent = pkf1->getParent().lock();
    if (parent && !parent->isBad())
      pEdgesKFs.push_back(parent);

    /// step3.3: 添加共视权重超过graph的边
    auto vEnssitialKFs = pkf1->getConnectedKfs(100);
    std::copy(vEnssitialKFs.begin(), vEnssitialKFs.end(), std::back_inserter(pEdgesKFs));

    for (auto &pkf2 : pEdgesKFs)
    {
      auto id2 = pkf2->getID();
      auto pair12 = std::make_pair(id1, id2);
      if (sAlreadyConnected.find(pair12) == sAlreadyConnected.end())
      {
        sAlreadyConnected.insert(pair12);
        auto e = new g2o::EdgeSim3();
        g2o::Sim3 S2w;
        if (mNoCorrectedG2oScw.find(pkf2) != noCorrectedEndIter)
          S2w = Converter::ConvertSim3G2o(mNoCorrectedG2oScw.at(pkf2));
        else
          S2w = vScwg[id2];

        e->setVertex(0, vpSim3Vertexes[id1]);
        e->setVertex(1, vpSim3Vertexes[id2]);
        e->setMeasurement(S2w * Sw1);
        e->setInformation(eInformation);
        e->setId(eID++);
        optimizer.addEdge(e);
      }
    }
  }
  optimizer.initializeOptimization();
  optimizer.optimize(20);

  /// step4: 矫正关键帧位姿
  auto g2oSwc0 = vpSim3Vertexes[0]->estimate().inverse();
  for (std::size_t idx = 0; idx <= nMaxID; ++idx)
  {
    auto &pVertex = vpSim3Vertexes[idx];
    if (!pVertex)
      continue;
    auto &pKf = vKeyFrames[idx];
    g2o::Sim3 g2oScwg = pVertex->estimate();
    Sim3Ret Scw = Converter::ConvertG2o2Sim3(g2oScwg * g2oSwc0);
    // Sim3Ret Scw = Converter::ConvertG2o2Sim3(g2oScwg);
    vG2oSwc[idx] = Scw.inv();
    pKf->setPose(Scw.mRqp, Scw.mtqp / Scw.mfS);
  }

  /// step5: 矫正地图点位置
  auto vAllMapPoints = mpMap->getAllMapPoints();
  for (const auto &pMp : vAllMapPoints)
  {
    if (!pMp || pMp->isBad())
      continue;
    KeyFrame::SharedPtr pCorrectKF;
    if (pMp->getLoopKF() == pCurrKf)
    {
      pCorrectKF = pMp->getLoopRefKF();
      pMp->setLoopRefKF(nullptr);
    }
    else
      pCorrectKF = pMp->getRefKF();
    if (!pCorrectKF || pCorrectKF->isBad())
      continue;
    cv::Mat p3dW = pMp->getPos();
    std::size_t kfID = pCorrectKF->getID();
    cv::Mat correctedP3dW = vG2oSwc[kfID] * (vScw[kfID] * p3dW);
    pMp->setPos(correctedP3dW);
    pMp->updateDescriptor();
    pMp->updateNormalAndDepth();
  }
  mpMap->setUpdate(true);
}

/**
 * @brief 优化全局BA
 * @details
 *      1. 这时的全局BA优化和局部建图线程的关键帧增加和删除是同步进行的
 *      2. 因此这里的vpKfs具有一定的滞后性，因此为了防止地图断裂，这里优化后只对关键帧的mTcwGBA做更新
 *      3. 这时，地图中具有mTcwGBA的，一定产生了优化，而没有mTcwGBA的一定没有参与优化
 * @param vpKfs         输入的关键帧
 * @param vpMps         输入的地图点
 * @param nIteration    输入的迭代次数
 * @param bStopFlag     输入的停止标志
 * @param bUseRk        输入的核函数标识
 */
void Optimizer::globalOptimization(const std::vector<KeyFramePtr> &vpKfs, const std::vector<MapPointPtr> &vpMps, int nIteration, bool *bStopFlag, bool bUseRk)
{
  g2o::SparseOptimizer optimizer;
  auto lm = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BSSE3>(std::make_unique<LSSE3Eigen>()));
  optimizer.setAlgorithm(lm);
  if (bStopFlag)
    optimizer.setForceStopFlag(bStopFlag);

  std::size_t nMaxKfID = KeyFrame::getMaxID();
  std::unordered_map<KeyFrame::SharedPtr, g2o::VertexSE3Expmap *> mpFrames;
  std::unordered_map<MapPoint::SharedPtr, g2o::VertexPointXYZ *> mpLandmarks;

  for (auto pkf : vpKfs)
  {
    if (!pkf || pkf->isBad())
      continue;
    cv::Mat Rcw, tcw;
    pkf->getPose(Rcw, tcw);
    auto pVertex = new g2o::VertexSE3Expmap();
    std::size_t nFId = pkf->getID();
    pVertex->setId(nFId);
    pVertex->setEstimate(Converter::ConvertTcw2SE3(Rcw, tcw));
    pVertex->setFixed(nFId == 0);
    optimizer.addVertex(pVertex);
    mpFrames.insert({pkf, pVertex});
  }

  int nEdgeID = 0;
  for (auto pMp : vpMps)
  {
    if (!pMp || pMp->isBad())
      continue;
    auto pVertex = new g2o::VertexPointXYZ();
    pVertex->setId(pMp->getID() + nMaxKfID + 1);
    pVertex->setMarginalized(true);
    pVertex->setEstimate(Converter::ConvertPw2Vector3(pMp->getPos()));
    optimizer.addVertex(pVertex);
    mpLandmarks.insert({pMp, pVertex});
    auto mObservations = pMp->getObservation();
    for (auto &obs : mObservations)
    {
      auto pKf = obs.first.lock();
      if (!pKf || pKf->isBad())
        continue;
      auto iter = mpFrames.find(pKf);
      if (iter == mpFrames.end())
        continue;
      auto pFrame = iter->second;
      auto &kp = pKf->getLeftKeyPoint(obs.second);
      auto &rightU = pKf->getRightU(obs.second);
      if (rightU > 0)
      {
        auto pEdgeStereo = new g2o::EdgeStereoSE3ProjectXYZ();
        pEdgeStereo->fx = Camera::mfFx;
        pEdgeStereo->fy = Camera::mfFy;
        pEdgeStereo->cx = Camera::mfCx;
        pEdgeStereo->cy = Camera::mfCy;
        pEdgeStereo->bf = Camera::mfBf;
        pEdgeStereo->setId(nEdgeID++);
        pEdgeStereo->setVertex(0, pVertex);
        pEdgeStereo->setVertex(1, pFrame);
        pEdgeStereo->setMeasurement(Eigen::Vector3d(kp.pt.x, kp.pt.y, rightU));
        pEdgeStereo->setInformation(Eigen::Matrix3d::Identity() * pKf->getScaledFactorInv2(kp.octave));
        if (bUseRk)
        {
          auto rk = new g2o::RobustKernelHuber();
          rk->setDelta(deltaStereo);
          pEdgeStereo->setRobustKernel(rk);
        }
        optimizer.addEdge(pEdgeStereo);
      }
      else
      {
        auto pEdgeMono = new g2o::EdgeSE3ProjectXYZ();
        pEdgeMono->fx = Camera::mfFx;
        pEdgeMono->fy = Camera::mfFy;
        pEdgeMono->cx = Camera::mfCx;
        pEdgeMono->cy = Camera::mfCy;
        pEdgeMono->setId(nEdgeID++);
        pEdgeMono->setVertex(0, pVertex);
        pEdgeMono->setVertex(1, pFrame);
        pEdgeMono->setMeasurement(Eigen::Vector2d(kp.pt.x, kp.pt.y));
        pEdgeMono->setInformation(Eigen::Matrix2d::Identity() * pKf->getScaledFactorInv2(kp.octave));
        if (bUseRk)
        {
          auto rk = new g2o::RobustKernelHuber();
          rk->setDelta(deltaMono);
          pEdgeMono->setRobustKernel(rk);
        }
        optimizer.addEdge(pEdgeMono);
      }
    }
  }
  optimizer.initializeOptimization(0);
  optimizer.optimize(nIteration);

  for (auto &item : mpFrames)
  {
    auto pkf = item.first;
    auto pVertex = item.second;
    pkf->mTcwGBA = Converter::ConvertSE32Tcw(pVertex->estimate());
  }

  for (auto &item : mpLandmarks)
  {
    auto pMp = item.first;
    auto pVertex = item.second;
    pMp->mPGBA = Converter::ConvertVector32Pw(pVertex->estimate());
  }
}

/// 将SIM3Ret类型转换为g2o::Sim3类型
g2o::Sim3 Converter::ConvertSim3G2o(const Sim3Ret &Scm)
{
  cv::Mat Rqp, tqp;
  g2o::Matrix3 Rqpg;
  g2o::Vector3 tqpg;
  Scm.mRqp.convertTo(Rqp, CV_64F);
  Scm.mtqp.convertTo(tqp, CV_64F);
  cv::cv2eigen(Rqp, Rqpg);
  cv::cv2eigen(tqp, tqpg);
  return g2o::Sim3(Rqpg, tqpg, (double)Scm.mfS);
}

/// 将g2o::Sim3类型转换为SIM3Ret类型
Sim3Ret Converter::ConvertG2o2Sim3(const g2o::Sim3 &Scm)
{
  cv::Mat Rqp(3, 3, CV_64F), tqp(3, 1, CV_64F);
  g2o::Matrix3 Rqpg = Scm.rotation().matrix();
  g2o::Vector3 tqpg = Scm.translation();
  cv::eigen2cv(Rqpg, Rqp);
  cv::eigen2cv(tqpg, tqp);
  Sim3Ret Sqp;
  Sqp.mfS = Scm.scale();
  Rqp.convertTo(Sqp.mRqp, CV_32F);
  tqp.convertTo(Sqp.mtqp, CV_32F);
  return Sqp;
}

/// 将Rcw的cv::Mat类型转换为Eigen::Quaternionf类型
Eigen::Quaternionf Converter::ConvertCV2Eigen(const cv::Mat &Rcw)
{
  Eigen::Matrix3f RcwE;
  cv::cv2eigen(Rcw, RcwE);
  Eigen::Quaternionf Qcw(RcwE);
  Qcw.normalize();
  return Qcw;
}

/// Optimizer的静态变量
float Optimizer::deltaMono = std::sqrt(5.991);   ///< 单目二自由度
float Optimizer::deltaStereo = std::sqrt(7.815); ///< 双目三自由度
float Optimizer::deltaSim3 = std::sqrt(9.210);   ///< Sim3二自由度
} // namespace ORB_SLAM2_ROS2