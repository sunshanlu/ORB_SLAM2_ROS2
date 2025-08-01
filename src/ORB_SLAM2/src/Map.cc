#include "ORB_SLAM2/Map.h"
#include "Map.pb.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"

namespace ORB_SLAM2_ROS2
{
/**
 * @brief 插入关键帧
 *
 * @param pKf   输入的要插入地图的关键帧
 * @param pMap  包装的this的共享指针
 */
void Map::insertKeyFrame(KeyFramePtr pKf, Map::SharedPtr pMap, bool bAddHas)
{
  {
    std::unique_lock<std::mutex> lock(mKfMutex);
    mspKeyFrames.insert(pKf);
    pKf->setMap(pMap);
  }
  setUpdate(true);
  if (bAddHas)
    mmKeyFrames.insert({pKf->getID(), pKf});
}

/**
 * @brief 插入地图点
 *
 * @param pMp   输入的要插入地图的地图点
 * @param pMap  包装的this的共享指针
 */
void Map::insertMapPoint(MapPointPtr pMp, Map::SharedPtr pMap)
{
  {
    std::unique_lock<std::mutex> lock(mMpMutex);
    if (pMp->isInMap())
      return;
    mspMapPoints.insert(pMp);
    pMp->setMap(pMap);
  }
  setUpdate(true);
}

/**
 * @brief 删除某个地图点
 *
 * @param pMp 输入的待删除的地图点
 */
void Map::eraseMapPoint(MapPointPtr pMp)
{
  {
    std::unique_lock<std::mutex> lock(mMpMutex);
    auto it = mspMapPoints.find(pMp);
    if (it == mspMapPoints.end())
      return;
    mspMapPoints.erase(pMp);
    pMp->setMapNull();
  }
  setUpdate(true);
}

/**
 * @brief 删除某个关键帧
 *
 * @param pKf 输入的待删除的关键帧
 */
void Map::eraseKeyFrame(KeyFramePtr pKf)
{
  {
    std::unique_lock<std::mutex> lock(mKfMutex);
    mspKeyFrames.erase(pKf);
    pKf->setMapNull();
  }
  setUpdate(true);
}

/**
 * @brief 将地图信息保存到文件中去
 *
 * @param DirPath 输入的文件夹路径，注意不需要以"/"结尾
 */
void Map::saveToTxtFile(const std::string &DirPath)
{
  std::ofstream ofsKf(DirPath + "/KeyFrames.txt");
  std::ofstream ofsMp(DirPath + "/MapPoints.txt");
  std::thread tkf(
      [this, &ofsKf]()
      {
        for (auto &pKf : this->mspKeyFrames)
        {
          if (!pKf || pKf->isBad())
            continue;
          ofsKf << *pKf;
        }
      });

  std::thread tmp(
      [this, &ofsMp]()
      {
        for (auto &pMp : this->mspMapPoints)
        {
          if (!pMp || pMp->isBad())
            continue;
          ofsMp << *pMp;
        }
      });
  tkf.join();
  tmp.join();
}

/**
 * @brief 在文件DirPath路径中，加载地图信息
 *
 * @param DirPath   输入的文件路径信息
 * @param pMap      输入的地图指针
 */
void Map::loadFromTxtFile(const std::string &DirPath, SharedPtr pMap)
{
  std::ifstream ifsKf(DirPath + "/KeyFrames.txt");
  std::ifstream ifsMp(DirPath + "/MapPoints.txt");
  std::map<std::size_t, std::pair<KeyFrame::SharedPtr, KeyFrameInfo>> mKeyFramesInfo;
  std::map<std::size_t, std::pair<MapPoint::SharedPtr, MapPointInfo>> mMapPointsInfo;

  /// 创建关键帧
  std::thread tKf(
      [&ifsKf, &pMap, &mKeyFramesInfo]()
      {
        while (1)
        {
          bool notEof = true;
          KeyFrameInfo kfInfo;
          auto pKf = KeyFrame::create(ifsKf, kfInfo, notEof);
          if (!notEof)
            break;
          pMap->insertKeyFrame(pKf, pMap, true);
          mKeyFramesInfo[pKf->getID()] = std::make_pair(pKf, kfInfo);
        }
        ifsKf.close();
      });

  /// 创建地图点
  std::thread tMp(
      [&ifsMp, &pMap, &mMapPointsInfo]()
      {
        while (1)
        {
          bool notEof = true;
          MapPointInfo mpInfo;
          auto pMp = MapPoint::create(ifsMp, mpInfo, notEof);
          if (!notEof)
            break;
          pMap->insertMapPoint(pMp, pMap);
          mMapPointsInfo[pMp->getID()] = std::make_pair(pMp, mpInfo);
        }
        ifsMp.close();
      });
  tKf.join();
  tMp.join();

  /// 处理关键帧的连接关系
  Map::processConnection(mKeyFramesInfo, mMapPointsInfo);
}

/**
 * @brief 获取仅跟踪状态下的参考关键帧
 * @details
 *      1. 无后端优化，基于恒速模型跟踪是及其不稳定的
 *      2. 这时候，仅跟踪状态下，需要以跟踪参考关键帧为主
 *      3. 由于仅跟踪状态下不能插入关键帧，因此需要对跟踪线程的参考关键帧进行更新
 *      4. 根据当前跟踪线程的参考关键帧，取前3个id和后3个id的关键帧作为更新候选
 *      5. 取相似对最高的参考关键帧作为当前跟踪线程的新关键帧
 * @param pTcurrFrame   输入的跟踪线程的当前关键帧
 * @param oldRefID      输入的跟踪线程的旧参考关键帧的id
 * @return Map::KeyFramePtr 输出的新的参考关键帧
 */
Map::KeyFramePtr Map::getTrackingRef(const FramePtr &pTcurrFrame, const std::size_t &oldRefID)
{
  double bestScore = 0;
  KeyFramePtr bestPkf = nullptr;
  std::size_t minId = std::max(0l, (long)oldRefID - 3);
  for (std::size_t cId = minId; cId < oldRefID + 3; ++cId)
  {
    auto iter = mmKeyFrames.find(cId);
    if (iter == mmKeyFrames.end())
      continue;
    KeyFramePtr pkf = iter->second;
    if (!pkf || pkf->isBad())
      continue;
    double score = pTcurrFrame->computeSimilarity(*iter->second);
    if (score > bestScore)
    {
      bestScore = score;
      bestPkf = pkf;
    }
  }
  return bestPkf;
}

/// 将地图保存到protobuf文件中
void Map::saveToProtobuf(const std::string &DirPath)
{
  orbslam2::MapData mapData;

  // 获取关键帧列表和地图点列表的指针
  auto *kfList = mapData.mutable_keyframes();
  auto *mpList = mapData.mutable_mappoints();

  // 设置关键帧全局信息
  kfList->set_next_id(KeyFrame::mnNextId);
  for (const auto &scale : KeyFrame::mvfScaledFactors)
    kfList->add_scale_factors(scale);

  // 序列化关键帧信息
  std::thread tkf(
      [this, kfList]()
      {
        for (const auto &pKf : this->mspKeyFrames)
        {
          if (!pKf || pKf->isBad())
            continue;
          auto *kfData = kfList->add_keyframes();
          pKf->serializeToProtobuf(*kfData);
        }
      });

  // 序列化地图点信息
  std::thread tmp(
      [this, mpList]()
      {
        for (const auto &pMp : this->mspMapPoints)
        {
          if (!pMp || pMp->isBad())
            continue;
          auto *mpData = mpList->add_mappoints();
          pMp->serializeToProtobuf(*mpData);
        }
      });

  tkf.join();
  tmp.join();

  std::string filePath = DirPath + "/map.pb";
  std::ofstream ofs(filePath, std::ios::binary);
  if (ofs.is_open())
  {
    mapData.SerializeToOstream(&ofs);
    ofs.close();
  }
}

/// 从protobuf文件中加载地图
void Map::loadFromProtobuf(const std::string &DirPath, SharedPtr pMap)
{
  // 从文件中读取protobuf数据
  std::string filePath = DirPath + "/map.pb";
  std::ifstream ifs(filePath, std::ios::binary);
  if (!ifs.is_open())
  {
    RCLCPP_ERROR(rclcpp::get_logger("ORB_SLAM2"), "无法打开protobuf地图文件: %s", filePath.c_str());
    return;
  }

  // 解析MapData
  orbslam2::MapData mapData;
  if (!mapData.ParseFromIstream(&ifs))
  {
    RCLCPP_ERROR(rclcpp::get_logger("ORB_SLAM2"), "解析protobuf地图文件失败");
    return;
  }

  std::map<std::size_t, std::pair<KeyFrame::SharedPtr, KeyFrameInfo>> mKeyFramesInfo;
  std::map<std::size_t, std::pair<MapPoint::SharedPtr, MapPointInfo>> mMapPointsInfo;

  // 设置关键帧全局信息
  const auto &kfList = mapData.keyframes();
  KeyFrame::mnNextId = kfList.next_id();
  KeyFrame::mvfScaledFactors.clear();
  for (int i = 0; i < kfList.scale_factors_size(); ++i)
    KeyFrame::mvfScaledFactors.push_back(kfList.scale_factors(i));

  // 创建关键帧
  std::thread tKf(
      [&kfList, &pMap, &mKeyFramesInfo]()
      {
        for (int i = 0; i < kfList.keyframes_size(); ++i)
        {
          const auto &kfData = kfList.keyframes(i);
          KeyFrameInfo kfInfo;

          // 创建关键帧并反序列化
          auto pKf = KeyFrame::create(kfData, kfInfo);
          pMap->insertKeyFrame(pKf, pMap, true);
          mKeyFramesInfo[pKf->getID()] = std::make_pair(pKf, kfInfo);
        }
      });

  // 创建地图点
  std::thread tMp(
      [&mapData, &pMap, &mMapPointsInfo]()
      {
        const auto &mpList = mapData.mappoints();
        for (int i = 0; i < mpList.mappoints_size(); ++i)
        {
          const auto &mpData = mpList.mappoints(i);
          MapPointInfo mpInfo;

          // 创建地图点并反序列化
          auto pMp = MapPoint::create(mpData, mpInfo);
          pMap->insertMapPoint(pMp, pMap);
          mMapPointsInfo[pMp->getID()] = std::make_pair(pMp, mpInfo);
        }
      });

  tKf.join();
  tMp.join();

  /// 处理关键帧的连接关系
  Map::processConnection(mKeyFramesInfo, mMapPointsInfo);
}

/// 处理地图中元素之间的连接关系
void Map::processConnection(std::map<std::size_t, std::pair<KeyFrame::SharedPtr, KeyFrameInfo>> &mKeyFramesInfo,
                            std::map<std::size_t, std::pair<MapPoint::SharedPtr, MapPointInfo>> &mMapPointsInfo)
{
  mKeyFramesInfo[0].first->setNotErased(true);
  for (auto &item : mKeyFramesInfo)
  {
    std::multimap<int, KeyFrame::SharedPtr, std::greater<int>> mOrderedC;
    auto &pkf = item.second.first;
    auto &kfInfo = item.second.second;
    for (auto &Connected : kfInfo.mmAllConnected)
    {
      auto &cID = Connected.first;
      auto &pkfConnected = mKeyFramesInfo[cID].first;
      pkf->mmConnectedKfs.insert({pkfConnected, Connected.second});
      mOrderedC.insert({Connected.second, pkfConnected});
    }
    for (auto &item : mOrderedC)
    {
      if (item.first > 15)
      {
        pkf->mlpConnectedKfs.push_back(item.second);
        pkf->mlnConnectedWeights.push_back(item.first);
      }
    }
    for (auto &childInfo : kfInfo.mvChildren)
    {
      auto &child = mKeyFramesInfo[childInfo].first;
      child->setParent(pkf);
      pkf->addChild(child);
    }
    for (auto &loopInfo : kfInfo.mvLoopEdges)
    {
      auto &loopEdge = mKeyFramesInfo[loopInfo].first;
      pkf->addLoopEdge(loopEdge);
    }
    for (std::size_t idx = 0; idx < kfInfo.mvMapPoints.size(); ++idx)
    {
      auto &mpInfo = kfInfo.mvMapPoints[idx];
      if (mpInfo == -1)
      {
        pkf->mvpMapPoints.push_back(nullptr);
        continue;
      }

      auto &pMp = mMapPointsInfo[mpInfo].first;
      pkf->mvpMapPoints.push_back(pMp);
      pMp->mObs.insert({pkf, idx});
    }
  }

  /// 处理地图点的观测关系
  for (auto &item : mMapPointsInfo)
  {
    auto &pMp = item.second.first;
    auto &pMpInfo = item.second.second;
    auto &refKf = mKeyFramesInfo[pMpInfo.mnRefKFid].first;
    pMp->mpRefKf = refKf;
    pMp->mnRefFeatID = pMpInfo.mnRefFeatID;
  }
}

} // namespace ORB_SLAM2_ROS2