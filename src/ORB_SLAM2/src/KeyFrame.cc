#include <sstream>

#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"

namespace ORB_SLAM2_ROS2
{

/**
 * @brief 获取与当前帧一阶相连的关键帧（大于给定连接权重阈值的）
 *
 * @param th 给定的连接权重阈值，只统计大于等于这个阈值的相连关键帧
 * @return std::vector<KeyFrame::SharedPtr> 输出一阶相连的关键帧
 */
std::vector<KeyFrame::SharedPtr> KeyFrame::getConnectedKfs(int th)
{
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
  while (wIt != wEnd)
  {
    if (*wIt >= th)
    {
      SharedPtr pkf = kfIt->lock();
      if (pkf && !pkf->isBad())
        connectedKfs.push_back(pkf);
    }
    else
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
KeyFrame::SharedPtr KeyFrame::updateConnections()
{
  {
    std::unique_lock<std::mutex> lock(mConnectedMutex);
    mlpConnectedKfs.clear();
    mlnConnectedWeights.clear();
    mmConnectedKfs.clear();
  }
  std::map<KeyFrame::SharedPtr, std::size_t> mapConnected;
  auto vpMapPoints = getMapPoints();
  for (auto &pMp : vpMapPoints)
  {
    if (!pMp || pMp->isBad())
      continue;
    MapPoint::Observations obs = pMp->getObservation();
    for (auto &obsItem : obs)
    {
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

    for (auto &item : mapConnected)
    {
      if (item.second > maxWeight)
      {
        maxWeight = item.second;
        pBestPkf = item.first;
      }
      if (item.second < 15)
        continue;
      weightKfs.insert(std::make_pair(item.second, item.first));
      mmConnectedKfs.insert(std::make_pair(item.first, item.second));
    }
    if (weightKfs.empty() && pBestPkf && !pBestPkf->isBad())
    {
      weightKfs.insert(std::make_pair(maxWeight, pBestPkf));
      mmConnectedKfs.insert(std::make_pair(pBestPkf, maxWeight));
    }
    for (auto &item : weightKfs)
    {
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
void KeyFrame::updateConnections(SharedPtr child)
{
  SharedPtr parent = child->updateConnections();
  if (!parent || parent->getID() > child->getID())
    return;
  if (child->isParent())
  {
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
std::vector<KeyFrame::SharedPtr> KeyFrame::getOrderedConnectedKfs(int nNum)
{
  decltype(mlpConnectedKfs) lpConnectedKfs;
  {
    std::unique_lock<std::mutex> lock(mConnectedMutex);
    lpConnectedKfs = mlpConnectedKfs;
  }
  std::vector<SharedPtr> connectedKfs;
  if (lpConnectedKfs.size() < nNum)
  {
    for (auto iter = lpConnectedKfs.begin(); iter != lpConnectedKfs.end(); ++iter)
    {
      SharedPtr pkf = iter->lock();
      if (pkf && !pkf->isBad())
        connectedKfs.push_back(pkf);
    }
    return connectedKfs;
  }
  auto iter = lpConnectedKfs.begin();
  int n = 0;
  for (auto &pkfWeak : lpConnectedKfs)
  {
    SharedPtr pkf = iter->lock();
    if (pkf && pkf->isBad())
    {
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
void KeyFrame::eraseConnection(SharedPtr pkf)
{
  std::unique_lock<std::mutex> lock(mConnectedMutex);
  auto iter = mmConnectedKfs.find(pkf);
  if (iter != mmConnectedKfs.end())
  {
    mmConnectedKfs.erase(iter);
    auto nIter = mlnConnectedWeights.begin();
    for (auto pIter = mlpConnectedKfs.begin(); pIter != mlpConnectedKfs.end(); ++pIter)
    {
      if (pIter->lock() == pkf)
      {
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
KeyFrame::WeakCompareFunc KeyFrame::weakCompare = [](KeyFrame::WeakPtr p1, KeyFrame::WeakPtr p2)
{
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

/**
 * @brief 从流中读取关键帧信息
 *
 * @param is        输入的输入流
 * @param kfInfo    输出的KeyFrameInfo
 */
bool KeyFrame::readFromStream(std::istream &is, KeyFrameInfo &kfInfo)
{
  std::string lineStr;
  std::stringstream streamStr;

  if (!mbScaled)
  {
    mbScaled = true;
    if (!getline(is, lineStr))
      return false;
    streamStr << lineStr;
    streamStr >> mnNextId;
    float scale;
    while (streamStr >> scale)
      mvfScaledFactors.push_back(scale);
  }

  streamStr.clear();
  if (!getline(is, lineStr))
    return false;
  streamStr << lineStr;

  // 读取关键帧基本信息
  streamStr >> mnId >> mfMaxU >> mfMaxV >> mfMinU >> mfMinV;

  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  while (1)
  {
    cv::KeyPoint kp;
    float rightU, depth;
    streamStr >> kp.pt.x >> kp.pt.y >> kp.octave >> kp.angle >> rightU >> depth;
    if (!streamStr)
      break;

    // 读取特征点信息
    mvFeatsLeft.push_back(kp);
    mvFeatsRightU.push_back(rightU);
    mvDepths.push_back(depth);
  }

  // 读取描述子信息
  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  while (1)
  {
    cv::Mat Desc(1, 32, CV_8U);
    for (int i = 0; i < 32; ++i)
    {
      int val;
      streamStr >> val;
      Desc.at<uchar>(i) = val;
    }
    if (!streamStr)
      break;
    mvLeftDescriptor.push_back(Desc);
  }

  // 填充词袋向量
  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  while (1)
  {
    unsigned int worldID;
    double worldValue;
    streamStr >> worldID >> worldValue;
    if (!streamStr)
      break;
    mBowVec.insert({worldID, worldValue});
  }

  // 填充特征向量
  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  while (1)
  {
    unsigned int NodeID;
    std::vector<unsigned int> FeaturesID;
    std::size_t num;
    streamStr >> NodeID >> num;
    for (std::size_t idx = 0; idx < num; ++idx)
    {
      unsigned int id;
      streamStr >> id;
      FeaturesID.push_back(id);
    }
    if (!streamStr)
      break;
    mFeatVec.insert({NodeID, FeaturesID});
  }

  // 读取关键帧位姿
  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  cv::Mat Rcw(3, 3, CV_32F), tcw(3, 1, CV_32F);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      streamStr >> Rcw.at<float>(i, j);
  for (int i = 0; i < 3; ++i)
    streamStr >> tcw.at<float>(i);
  setPose(Rcw, tcw);

  // 关键帧之间的共视关系
  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  while (1)
  {
    int weight;
    std::size_t id;
    streamStr >> id >> weight;
    if (!streamStr)
      break;
    kfInfo.mmAllConnected.insert({id, weight});
  }

  // 读取子关键帧
  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  while (1)
  {
    std::size_t id;
    streamStr >> id;
    if (!streamStr)
      break;
    kfInfo.mvChildren.push_back(id);
  }

  // 读取回环关键帧
  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  while (1)
  {
    std::size_t id;
    streamStr >> id;
    if (!streamStr)
      break;
    kfInfo.mvLoopEdges.push_back(id);
  }

  // 读取地图点
  streamStr.clear();
  getline(is, lineStr);
  streamStr << lineStr;
  while (1)
  {
    long id;
    streamStr >> id;
    if (!streamStr)
      break;
    kfInfo.mvMapPoints.push_back(id);
  }
  return true;
}

/**
 * @brief 关键帧的信息保存
 *
 * @param os 输入的输出流
 * @param kf 输入的关键帧
 * @return std::ostream&
 */
std::ostream &operator<<(std::ostream &os, const KeyFrame &kf)
{
  // std::vector<std::size_t> vConnected;      ///< mlpConnectedKfs
  // std::vector<int> vWeights;                ///< mlnConnectedWeights
  std::map<std::size_t, int> mAllConnected; ///< mmConnectedKfs
  std::vector<std::size_t> vChildren;       ///< mspChildren
  std::vector<std::size_t> vLoopEdges;      ///< mvpLoopEdges
  std::vector<long> vMapPoints;             ///< mvpMapPoints

  // 将关联关系保存到mAllConnected中<id, weight>
  for (auto &it : kf.mmConnectedKfs)
  {
    auto pKf = it.first.lock();
    if (pKf && !pKf->isBad())
      mAllConnected.insert({pKf->getID(), it.second});
  }
  auto kfIter = kf.mlpConnectedKfs.begin();
  auto wIter = kf.mlnConnectedWeights.begin();

  // 连接关键帧和连接权重，在mmConnectedKfs中已经有描述了
  // while (kfIter != kf.mlpConnectedKfs.end() && wIter != kf.mlnConnectedWeights.end())
  // {
  //   auto pKf = (*kfIter).lock();
  //   if (pKf && !pKf->isBad())
  //   {
  //     vConnected.push_back(pKf->getID());
  //     vWeights.push_back(*wIter);
  //   }
  //   ++kfIter;
  //   ++wIter;
  // }

  // 维护关键帧的儿子节点
  for (auto &ChildWeak : kf.mspChildren)
  {
    auto child = ChildWeak.lock();
    if (child && !child->isBad())
      vChildren.push_back(child->getID());
  }

  // 维护关键帧的回环节点
  for (auto &pLoopWeak : kf.mvpLoopEdges)
  {
    auto pLoopKf = pLoopWeak.lock();
    if (pLoopKf && !pLoopKf->isBad())
      vLoopEdges.push_back(pLoopKf->getID());
  }

  // 维护关键帧的地图点id，索引id要对齐保证和关键点之间的索引关系
  for (auto &pMp : kf.mvpMapPoints)
  {
    if (pMp && !pMp->isBad())
      vMapPoints.push_back(pMp->getID());
    else
      vMapPoints.push_back(-1);
  }

  // 保存金字塔缩放比例，只保存一次
  if (kf.mbScaled)
  {
    os << kf.mnNextId << " ";
    for (auto &scale : kf.mvfScaledFactors)
    {
      os << scale << " ";
    }
    kf.mbScaled = false;
    os << std::endl;
  }

  // 保存关键帧id，去畸变后的像素范围
  os << kf.mnId << " " << kf.mfMaxU << " " << kf.mfMaxV << " " << kf.mfMinU << " " << kf.mfMinV << std::endl;

  // 保存关键点信息，像素位置，金字塔层级、角度、右图特征点u坐标、深度
  std::size_t nKp = kf.mvFeatsLeft.size();
  for (std::size_t idx = 0; idx < nKp; ++idx)
  {
    auto &kp = kf.mvFeatsLeft[idx];
    os << kp.pt.x << " " << kp.pt.y << " " << kp.octave << " " << kp.angle << " " << kf.mvFeatsRightU[idx] << " " << kf.mvDepths[idx] << " ";
  }
  os << std::endl;

  // 保存关键点描述子
  for (std::size_t idx = 0; idx < nKp; ++idx)
  {
    cv::Mat Desc = kf.mvLeftDescriptor[idx];
    for (std::size_t jdx = 0; jdx < 32; ++jdx)
      os << (int)Desc.at<uchar>(jdx) << " ";
  }
  os << std::endl;

  // 保存关键点的视觉单词
  for (auto &item : kf.mBowVec)
    os << item.first << " " << item.second << " ";
  os << std::endl;

  // 保存关键点的词袋视觉特征向量
  for (auto &item : kf.mFeatVec)
  {
    os << item.first << " " << item.second.size() << " ";
    for (auto &id : item.second)
      os << id << " ";
  }
  os << std::endl;

  // 保存关键帧的位姿
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      os << kf.mRcw.at<float>(i, j) << " ";
  for (int i = 0; i < 3; ++i)
    os << kf.mtcw.at<float>(i) << " ";
  os << std::endl;

  // 保存关键帧关联关系
  for (auto &item : mAllConnected)
    os << item.first << " " << item.second << " ";
  os << std::endl;

  // 保存子关键帧
  for (auto &item : vChildren)
    os << item << " ";
  os << std::endl;

  // 保存回环关键帧
  for (auto &item : vLoopEdges)
    os << item << " ";
  os << std::endl;

  // 保存关键帧对应地图点id
  for (auto &item : vMapPoints)
    os << item << " ";
  os << std::endl;

  return os;
}

/// 基于输入流的构造函数
KeyFrame::KeyFrame(std::istream &ifs, KeyFrameInfo &kfInfo, bool &notEof)
    : mspChildren(weakCompare)
    , mmConnectedKfs(weakCompare)
{
  mTcw = cv::Mat::eye(4, 4, CV_32F);
  mTwc = cv::Mat::eye(4, 4, CV_32F);
  notEof = readFromStream(ifs, kfInfo);
  if (!notEof)
    return;
  VirtualFrame::initGrid();
}

/**
 * @brief 基于protobuf,实现keyframe的序列化操作，用于保存数据
 *
 * @param data 输出的关键帧的protobuf数据
 */
void KeyFrame::serializeToProtobuf(orbslam2::KeyFrameData &data) const
{
  // todo 需要设置全局信息

  data.set_id(mnId);
  data.set_max_u(mfMaxU);
  data.set_max_v(mfMaxV);
  data.set_min_u(mfMinU);
  data.set_min_v(mfMinV);

  // 特征点信息，右图像素坐标和深度信息
  std::size_t nKp = mvFeatsLeft.size();
  for (std::size_t idx = 0; idx < nKp; ++idx)
  {
    const auto &kp = mvFeatsLeft[idx];
    auto *keypoint = data.add_keypoints();
    keypoint->set_x(kp.pt.x);
    keypoint->set_y(kp.pt.y);
    keypoint->set_octave(kp.octave);
    keypoint->set_angle(kp.angle);
    data.add_right_u(mvFeatsRightU[idx]);
    data.add_depths(mvDepths[idx]);
  }

  // 保存关键点描述子
  for (const auto &desc : mvLeftDescriptor)
  {
    auto *descriptor = data.add_descriptors();
    descriptor->set_data(desc.data, 32);
  }

  // 保存词袋向量
  auto *bowVec = data.mutable_bow_vector();
  for (const auto &item : mBowVec)
  {
    (*bowVec->mutable_words())[item.first] = item.second;
  }

  // 保存词袋特征向量
  auto *featVec = data.mutable_feature_vector();
  for (const auto &item : mFeatVec)
  {
    auto *node = featVec->add_nodes();
    node->set_node_id(item.first);
    for (const auto &featureId : item.second)
      node->add_feature_ids(featureId);
  }

  // 保存关键帧位姿
  auto *pose = data.mutable_pose();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      pose->add_rotation(mRcw.at<float>(i, j));
  for (int i = 0; i < 3; ++i)
    pose->add_translation(mtcw.at<float>(i));

  // 保存关键帧关联关系
  for (const auto &connected : mmConnectedKfs)
  {
    auto pkf = connected.first.lock();
    if (pkf && !pkf->isBad())
    {
      auto *conn = data.add_connected_kfs();
      conn->set_id(pkf->getID());
      conn->set_weight(connected.second);
    }
  }

  // 保存子关键帧
  for (const auto &childWeak : mspChildren)
  {
    auto child = childWeak.lock();
    if (child && !child->isBad())
      data.add_children_ids(child->getID());
  }

  // 保存回环关键帧
  for (const auto &loopEdgeWeak : mvpLoopEdges)
  {
    auto loopEdge = loopEdgeWeak.lock();
    if (loopEdge && !loopEdge->isBad())
      data.add_loop_edges(loopEdge->getID());
  }

  // 保存关键帧对应地图点id
  for (const auto &mp : mvpMapPoints)
  {
    if (mp && !mp->isBad())
      data.add_map_points(mp->getID());
    else
      data.add_map_points(-1);
  }
}

bool KeyFrame::deserializeFromProtobuf(const orbslam2::KeyFrameData &data, KeyFrameInfo &kfInfo)
{
  // todo 注意，这里需要更新keyframe的静态变量，scale和nextid

  // 读取关键帧基本信息
  mnId = data.id();
  mfMaxU = data.max_u();
  mfMaxV = data.max_v();
  mfMinU = data.min_u();
  mfMinV = data.min_v();

  mvFeatsLeft.clear();
  mvFeatsRightU.clear();
  mvDepths.clear();

  int nKp = data.keypoints_size();
  mvFeatsLeft.reserve(nKp);
  mvFeatsRightU.reserve(nKp);
  mvDepths.reserve(nKp);

  // 读取特征点信息
  for (int idx = 0; idx < nKp; ++idx)
  {
    const auto &kp_data = data.keypoints(idx);
    cv::KeyPoint kp;
    kp.pt.x = kp_data.x();
    kp.pt.y = kp_data.y();
    kp.octave = kp_data.octave();
    kp.angle = kp_data.angle();
    mvFeatsLeft.push_back(kp);
    mvFeatsRightU.push_back(data.right_u(idx));
    mvDepths.push_back(data.depths(idx));
  }

  // 读取描述子信息
  mvLeftDescriptor.clear();
  mvLeftDescriptor.reserve(data.descriptors_size());
  for (int idx = 0; idx < data.descriptors_size(); ++idx)
  {
    const auto &desc_data = data.descriptors(idx);
    cv::Mat descriptor = cv::Mat(1, 32, CV_8U);
    if (desc_data.data().size() >= 32)
      memcpy(descriptor.data, desc_data.data().c_str(), 32);
    mvLeftDescriptor.push_back(descriptor);
  }

  // 读取词袋向量
  mBowVec.clear();
  const auto &bow_data = data.bow_vector();
  for (const auto &word : bow_data.words())
    mBowVec.insert({word.first, word.second});

  // 填充词袋特征向量
  mFeatVec.clear();
  const auto &feat_data = data.feature_vector();
  for (int i = 0; i < feat_data.nodes_size(); ++i)
  {
    const auto &node = feat_data.nodes(i);
    std::vector<unsigned int> feature_ids;
    feature_ids.reserve(node.feature_ids_size());

    for (int j = 0; j < node.feature_ids_size(); ++j)
      feature_ids.push_back(node.feature_ids(j));

    mFeatVec.insert({node.node_id(), feature_ids});
  }

  // 读取关键帧位姿
  const auto &pose_data = data.pose();
  cv::Mat Rcw = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat tcw = cv::Mat::zeros(3, 1, CV_32F);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      Rcw.at<float>(i, j) = pose_data.rotation(i * 3 + j);
  for (int i = 0; i < 3; ++i)
    tcw.at<float>(i) = pose_data.translation(i);
  setPose(Rcw, tcw);

  // 读取关键帧之间的共视关系
  kfInfo.mmAllConnected.clear();
  for (int i = 0; i < data.connected_kfs_size(); ++i)
  {
    const auto &conn = data.connected_kfs(i);
    kfInfo.mmAllConnected.insert({conn.id(), conn.weight()});
  }

  // 读取子关键帧
  kfInfo.mvChildren.clear();
  kfInfo.mvChildren.reserve(data.children_ids_size());
  for (int i = 0; i < data.children_ids_size(); ++i)
    kfInfo.mvChildren.push_back(data.children_ids(i));

  // 读取回环关键帧
  kfInfo.mvLoopEdges.clear();
  kfInfo.mvLoopEdges.reserve(data.loop_edges_size());
  for (int i = 0; i < data.loop_edges_size(); ++i)
    kfInfo.mvLoopEdges.push_back(data.loop_edges(i));

  // 读取地图点
  kfInfo.mvMapPoints.clear();
  kfInfo.mvMapPoints.reserve(data.map_points_size());
  for (int i = 0; i < data.map_points_size(); ++i)
    kfInfo.mvMapPoints.push_back(data.map_points(i));

  return true;
}

/// 基于protobuf的构造函数
KeyFrame::KeyFrame(const orbslam2::KeyFrameData &data, KeyFrameInfo &kfInfo)
    : mspChildren(weakCompare)
    , mmConnectedKfs(weakCompare)
{
  mTcw = cv::Mat::eye(4, 4, CV_32F);
  mTwc = cv::Mat::eye(4, 4, CV_32F);

  deserializeFromProtobuf(data, kfInfo);
  VirtualFrame::initGrid();
}

} // namespace ORB_SLAM2_ROS2