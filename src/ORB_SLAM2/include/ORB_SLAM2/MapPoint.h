#pragma once

#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "KeyFrame.h"
namespace ORB_SLAM2_ROS2
{
class Map;

struct MapPointInfo
{
  std::size_t mnRefKFid;
  int mnRefFeatID;
};

class MapPoint
{
  friend class Map;
  friend std::ostream &operator<<(std::ostream &os, const MapPoint &mp);

public:
  typedef std::shared_ptr<MapPoint> SharedPtr;
  typedef std::weak_ptr<Map> MapWeakPtr;
  typedef std::shared_ptr<Map> MapPtr;
  typedef std::weak_ptr<KeyFrame> KeyFrameWeakPtr;
  typedef std::shared_ptr<KeyFrame> KeyFramePtr;
  typedef std::shared_ptr<VirtualFrame> VirtualFramePtr;
  typedef std::map<KeyFrameWeakPtr, std::size_t, KeyFrame::WeakCompareFunc> Observations;

  /// 从流中读取地图信息
  bool readFromStream(std::istream &is, MapPointInfo &info);

  /// 检查地图点是否合格
  bool checkMapPoint(KeyFramePtr pkf1, KeyFramePtr pkf2, const std::size_t &nFeatId1, const std::size_t &nFeatId2);

  /// 由关键帧构造
  static MapPoint::SharedPtr create(cv::Mat mP3dW, KeyFramePtr pKf, std::size_t nObs);

  /// 由普通帧构造
  static MapPoint::SharedPtr create(cv::Mat mP3dW);

  /// 基于输入流的构造
  static MapPoint::SharedPtr create(std::istream &ifs, MapPointInfo &info, bool &notEof)
  {
    std::size_t nMaxID = 0;
    auto ptr = MapPoint::SharedPtr(new MapPoint(ifs, info, notEof));
    if (ptr->getID() > nMaxID)
    {
      mnNextId = nMaxID + 1;
    }
    return ptr;
  }

  /// 地图点添加观测信息
  void addObservation(KeyFramePtr pkf, std::size_t featId);

  std::size_t getID() { return mId; }

  /// 获取观测信息
  Observations getObservation()
  {
    std::unique_lock<std::mutex> lock(mObsMutex);
    return mObs;
  }
  const Observations getObservation() const
  {
    std::unique_lock<std::mutex> lock(mObsMutex);
    return mObs;
  }

  /// 获取观测信息中可靠的数目
  int getObsNum();

  /// 地图点设置地图
  void setMap(MapWeakPtr pMap)
  {
    mbIsInMap = true;
    mpMap = pMap;
  }

  /// 判断this地图点是否在某个关键帧中
  bool isInKeyFrame(KeyFramePtr pkf);

  /// 判断地图点是否是bad
  bool isBad() const;

  /// 判断地图点是否在map中
  bool isInMap() const { return mbIsInMap; };

  /// 获取观测方向
  cv::Mat getViewDirection() const
  {
    std::unique_lock<std::mutex> lock(mMutexView);
    return mViewDirection.clone();
  }

  /// 设置观测方向
  void setViewDirection(cv::Mat viewDirection)
  {
    std::unique_lock<std::mutex> lock(mMutexView);
    viewDirection.copyTo(mViewDirection);
  }

  /// 返回地图点位置
  cv::Mat getPos() const
  {
    std::unique_lock<std::mutex> lock(mPosMutex);
    return mPoint3d.clone();
  }

  /// 设置bad flag
  void setBad()
  {
    std::unique_lock<std::mutex> lock(mBadMutex);
    mbIsBad = true;
  }

  /// 设置地图点位置
  void setPos(cv::Mat point3d)
  {
    std::unique_lock<std::mutex> lock(mPosMutex);
    point3d.copyTo(mPoint3d);
  }

  /// 设置最大距离和最小距离
  void setDistance(KeyFramePtr pRefKf, std::size_t nFeatId);

  /// 获取地图点的最大和最小距离
  void getDistance(float &nMax, float &nMin) const
  {
    std::unique_lock<std::mutex> lock(mMutexDis);
    nMax = mnMaxDistance;
    nMin = mnMinDistance;
  }

  /// 判断d是否在最大距离和最小距离之间
  bool isGoodDistance(const float &distance) const
  {
    std::unique_lock<std::mutex> lock(mMutexDis);
    if (distance < mnMaxDistance && distance > mnMinDistance)
      return true;
    return false;
  }

  /// 获取是否在局部地图中
  bool isLocalMp() const { return mbIsLocalMp; }

  /// 设置地图点的参考关键帧
  void setRefKF(KeyFramePtr pRefkf, std::size_t nFeatId)
  {
    std::unique_lock<std::mutex> lock(mMutexRefKF);
    mpRefKf = pRefkf;
    mnRefFeatID = nFeatId;
  }

  /// 获取地图点的参考关键帧
  void getRefKF(KeyFramePtr &pRefKf, std::size_t &nFeatId)
  {
    std::unique_lock<std::mutex> lock(mMutexRefKF);
    pRefKf = mpRefKf.lock();
    nFeatId = mnRefFeatID;
  }

  /// 设置地图点的代表性描述子
  void setDesc(cv::Mat desc)
  {
    std::unique_lock<std::mutex> lock(mMutexDesc);
    desc.copyTo(mDescriptor);
  }

  /// 设置是否在局部地图中
  void setLocalMp(bool bIsLocalMp) { mbIsLocalMp = bIsLocalMp; }

  /// 预测金字塔层级
  int predictLevel(float distance) const;

  /// 跟踪线程初始化专用函数（由普通地图点添加属性到特殊地图点上）
  void addAttriInit(KeyFramePtr pRefKf, std::size_t nFeatId);

  /// 判断某一位姿是否在共视范围内
  bool isInVision(VirtualFramePtr pFrame, float &vecDistance, cv::Point2f &uv, float &cosTheta);

  /// 获取地图点的代表描述子
  cv::Mat getDesc() const
  {
    std::unique_lock<std::mutex> lock(mMutexDesc);
    return mDescriptor.clone();
  }

  /// 设置nullptr在map上
  void setMapNull()
  {
    mpMap.reset();
    mbIsInMap = false;
  }

  /// 添加跟踪的匹配信息
  void addMatchInTrack()
  {
    std::unique_lock<std::mutex> lock(mTrackMutex);
    ++mnMatchesInTrack;
  }

  /// 添加跟踪的优化内点信息
  void addInlierInTrack()
  {
    std::unique_lock<std::mutex> lock(mTrackMutex);
    ++mnInliersInTrack;
  }

  /// 获取跟踪的匹配信息
  int getMatchInTrack()
  {
    std::unique_lock<std::mutex> lock(mTrackMutex);
    return mnMatchesInTrack;
  }

  /// 获取跟踪的优化内点信息
  int getinlierInTrack()
  {
    std::unique_lock<std::mutex> lock(mTrackMutex);
    return mnInliersInTrack;
  }

  /// 更新跟踪的匹配信息和内点优化信息
  void updateTrackParam(SharedPtr pMp)
  {
    std::unique_lock<std::mutex> lock(mTrackMutex);
    mnMatchesInTrack += pMp->getMatchInTrack();
    mnInliersInTrack += pMp->getinlierInTrack();
  }

  /// 返回地图点在跟踪过程中的重要程度
  float scoreInTrack()
  {
    std::unique_lock<std::mutex> lock(mTrackMutex);
    if (!mnMatchesInTrack)
      return 0;
    return (float)mnInliersInTrack / mnMatchesInTrack;
  }

  /// 更新当前关键帧的描述子
  void updateDescriptor();

  /// 更新当前关键帧的深度和平均观测方向(如果参考关键帧被删除，同样会进行更新)
  void updateNormalAndDepth();

  /// 获取观测中有效的关键帧和索引pair
  std::vector<std::pair<KeyFramePtr, std::size_t>> getPostiveObs();

  /// 计算指定行向量的4分位数
  float computeMedian(const cv::Mat &vec);

  /// 替换地图点
  static void replace(SharedPtr pMp1, SharedPtr pMp2, MapPtr pMap);

  /// 清除观测（需要返回观测信息）
  Observations clearObversition()
  {
    std::unique_lock<std::mutex> lock(mObsMutex);
    auto obs = mObs;
    mObs.clear();
    return obs;
  }

  /// 删除指定的观测信息
  void eraseObservetion(KeyFramePtr pkf, bool checkRef = true);

  /// 获取参考关键帧
  KeyFramePtr getRefKF();

  /// 获取回环矫正时刻关键帧
  KeyFramePtr getLoopKF();

  /// 获取回环矫正参考关键帧
  KeyFramePtr getLoopRefKF();

  /// 设置回环矫正时刻关键帧
  void setLoopKF(KeyFramePtr pkf);

  /// 设置回环矫正参考关键帧
  void setLoopRefKF(KeyFramePtr refKF);

  /// 获取地图点的最大id
  static const std::size_t getMaxID() { return mnNextId - 1; }

private:
  /// 用于关键帧构造的地图点
  MapPoint(cv::Mat mP3d, KeyFramePtr pRefKf, std::size_t nObs);

  /// 用于普通帧构造的临时地图点
  MapPoint(cv::Mat mP3d);

  /// 基于输入流的构造函数
  MapPoint(std::istream &ifs, MapPointInfo &info, bool &notEof);

  static unsigned int mnNextId;   ///< 下一个地图点的id
  unsigned int mId;               ///< this的地图点id
  cv::Mat mPoint3d;               ///< this的地图点3d坐标（世界坐标系下）
  bool mbIsInMap = false;         ///< 是否在地图中
  Observations mObs;              ///< 地图点的观测
  MapWeakPtr mpMap;               ///< 地图
  bool mbIsBad = false;           ///< 地图点是否是bad
  mutable std::mutex mBadMutex;   ///< 维护mbIsBad的互斥锁
  mutable std::mutex mPosMutex;   ///< 维护mPoint3d的互斥锁
  mutable std::mutex mObsMutex;   ///< 维护mObs的互斥锁
  mutable std::mutex mTrackMutex; ///< 维护跟踪参数的互斥锁
  // mutable std::mutex mRefKFMutex; ///< 维护参考关键帧的互斥锁
  cv::Mat mDescriptor;            ///< 地图点的代表描述子
  cv::Mat mViewDirection;         ///< 地图点的观测方向（光心->地图点）
  float mnMaxDistance;            ///< 最大匹配距离
  float mnMinDistance;            ///< 最小匹配距离
  KeyFrameWeakPtr mpRefKf;        ///< 地图点的参考关键帧
  std::size_t mnRefFeatID;        ///< 参考关键帧的ORB特征点索引
  int mnMatchesInTrack = 0;       ///< 在跟踪过程中，被匹配成功的
  int mnInliersInTrack = 0;       ///< 在跟踪过程中，经过优化后还是内点的
  KeyFrameWeakPtr mpLoopTime;     ///< 在回环闭合过程中矫正位置时刻的关键帧
  KeyFrameWeakPtr mpLoopRefKF;    ///< 在回环闭合过程中矫正参考关键帧
  mutable std::mutex mMutexRefKF; ///< 参考关键帧和2d点位置的互斥锁
  mutable std::mutex mMutexDis;   ///< distance的互斥锁
  mutable std::mutex mMutexView;  ///< 观测向量的互斥锁
  mutable std::mutex mMutexDesc;  ///< 描述子的互斥锁

public:
  bool mbIsLocalMp = false; ///< 是否在跟踪线程的局部地图中
  cv::Mat mPGBA;            ///< 全局BA优化后的位置
};

/// 将地图点信息输出到流中
std::ostream &operator<<(std::ostream &os, const MapPoint &mp);

} // namespace ORB_SLAM2_ROS2