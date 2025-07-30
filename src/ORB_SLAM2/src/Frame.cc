#include <chrono>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Error.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"

namespace ORB_SLAM2_ROS2
{

/**
 * @brief 展示双目图像的匹配结果
 *
 */
void Frame::showStereoMatches() const
{
  cv::Mat showImage;
  std::vector<cv::Mat> imgs{mLeftIm, mRightIm};
  cv::hconcat(imgs, showImage);
  cv::cvtColor(showImage, showImage, cv::COLOR_GRAY2BGR);
  std::vector<cv::KeyPoint> rightKps, leftKps;
  for (std::size_t i = 0; i < mvFeatsLeft.size(); ++i)
  {
    const auto &rightU = mvFeatsRightU[i];
    const auto &depth = mvDepths[i];
    if (rightU == -1)
    {
      continue;
    }
    const auto &lkp = mvFeatsLeft[i];
    cv::KeyPoint rkp;
    rkp.pt.x = rightU + mLeftIm.cols;
    rkp.pt.y = lkp.pt.y;
    cv::line(showImage, lkp.pt, rkp.pt, cv::Scalar(255, 0, 0));
    rightKps.push_back(rkp);
    leftKps.push_back(lkp);
    /// 输出左图，右图和深度值
    std::cout << lkp.pt.x << std::endl;
    std::cout << lkp.pt.y << std::endl;
    std::cout << rightU << std::endl;
    std::cout << depth << std::endl;
    std::cout << "===========================" << std::endl;
  }
  cv::drawKeypoints(showImage, leftKps, showImage, cv::Scalar(0, 255, 0));
  cv::drawKeypoints(showImage, rightKps, showImage, cv::Scalar(0, 0, 255));
  cv::imshow("showImage", showImage);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

/// 初始化网格
void VirtualFrame::initGrid()
{
  int rows = cvCeil((float)(mfMaxV - mfMinV) / mnGridHeight);
  int cols = cvCeil((float)(mfMaxU - mfMinU) / mnGridWidth);

  mGrids.resize(rows);
  for (auto &grid : mGrids)
    grid.resize(cols);

  for (std::size_t idx = 0; idx < mvFeatsLeft.size(); ++idx)
  {
    auto &feat = mvFeatsLeft[idx];
    std::size_t rowIdx = cvFloor(feat.pt.y / mnGridHeight);
    std::size_t colIdx = cvFloor(feat.pt.x / mnGridWidth);
    mGrids[rowIdx][colIdx].push_back(idx);
  }
}

/**
 * @brief 普通帧的构造函数
 * @details
 *      1. 多线程特征提取
 *      2. 金字塔缩放因子初始化
 *      3. 地图点长度初始化
 *      4. 特征点网格库建立
 * @param leftImg   左图图像
 * @param rightImg  右图图像
 * @param nFeatures 需要的特征点数
 * @param briefFp   BRIEF模板文件路径
 * @param maxThresh FAST检测的最大阈值
 * @param minThresh FAST检测的最小阈值
 */
Frame::Frame(cv::Mat leftImg, cv::Mat rightImg, int nFeatures, const std::string &briefFp, int maxThresh, int minThresh, VocabPtr pVoc, int nLevels,
             float scale)
    : mLeftIm(leftImg)
    , mRightIm(rightImg)
    , VirtualFrame(leftImg.cols, leftImg.rows, pVoc)
{
  mpExtractorLeft = std::make_shared<ORBExtractor>(mLeftIm, nFeatures, nLevels, scale, briefFp, maxThresh, minThresh);
  mpExtractorRight = std::make_shared<ORBExtractor>(mRightIm, nFeatures, nLevels, scale, briefFp, maxThresh, minThresh);

  if (!mbScaled)
  {
    mvfScaledFactors = mpExtractorLeft->getScaledFactors();
    mbScaled = true;
  }

  std::thread leftThread(std::bind(&ORBExtractor::extract, mpExtractorLeft.get(), std::ref(mvFeatsLeft), std::ref(mvLeftDescriptor)));
  std::thread rightThread(std::bind(&ORBExtractor::extract, mpExtractorRight.get(), std::ref(mvFeatsRight), std::ref(mRightDescriptor)));
  if (!leftThread.joinable() || !rightThread.joinable())
    throw ThreadError("普通帧提取特征点时线程不可joinable");
  leftThread.join();
  rightThread.join();
  Camera::undistortPoints(mvFeatsLeft);

  mvpMapPoints.resize(mvFeatsLeft.size(), nullptr);
  VirtualFrame::initGrid();
  mnID = mnNextID;
}

/**
 * @brief Construct a new Frame object
 *
 * @param colorImg  彩色图像
 * @param depthImg  深度图像 (要求以m为单位)
 * @param nFeatures 需要提取的特征数
 * @param briefFp   BRIEF模版文件路径
 * @param maxThresh FAST检测的最大阈值
 * @param minThresh FAST检测的最小阈值
 * @param pVoc      词袋模型指针
 * @param dScale    深度图像缩放尺度
 */
Frame::Frame(cv::Mat colorImg, cv::Mat depthImg, int nFeatures, const std::string &briefFp, int maxThresh, int minThresh, VocabPtr pVoc, float dScale,
             int nLevels, float scale)
    : mLeftIm(colorImg)
    , VirtualFrame(colorImg.cols, colorImg.rows, pVoc)
{
  depthImg.convertTo(depthImg, CV_32F);
  depthImg /= dScale;
  mpExtractorLeft = std::make_shared<ORBExtractor>(mLeftIm, nFeatures, nLevels, scale, briefFp, maxThresh, minThresh);
  if (!mbScaled)
  {
    mvfScaledFactors = mpExtractorLeft->getScaledFactors();
    mbScaled = true;
  }
  mpExtractorLeft->extract(mvFeatsLeft, mvLeftDescriptor);
  decltype(mvFeatsLeft) vFeatsLeftCopy(mvFeatsLeft);
  Camera::undistortPoints(mvFeatsLeft);
  auto FeaturesN = mvFeatsLeft.size();
  mvpMapPoints.resize(FeaturesN, nullptr);
  mvDepths.resize(FeaturesN, -1);
  mvFeatsRightU.resize(FeaturesN, -1);
  VirtualFrame::initGrid();
  mnID = mnNextID;

  for (std::size_t idx = 0; idx < FeaturesN; ++idx)
  {
    const cv::KeyPoint &kp = vFeatsLeftCopy[idx];
    const cv::KeyPoint &kpU = mvFeatsLeft[idx];
    const float d = depthImg.at<float>(kp.pt.y, kp.pt.x);
    if (d > 0)
    {
      mvDepths[idx] = d;
      mvFeatsRightU[idx] = kpU.pt.x - Camera::mfBf / d;
    }
  }
}

/**
 * @brief 设置指定位置的地图点
 *
 * @param idx 输入的指定位置索引
 * @param pMp 输入的地图点
 */
void VirtualFrame::setMapPoint(int idx, MapPointPtr pMp)
{
  std::unique_lock<std::mutex> lock(mMutexMapPoints);
  mvpMapPoints[idx] = pMp;
}

/**
 * @brief 初始化地图点
 * @details
 *      1. 普通帧没有创建地图点的权限，只是利用普通帧的信息进行地图点计算
 * @param mapPoints 输出的地图点信息
 */
int Frame::unProject(std::vector<MapPoint::SharedPtr> &mapPoints)
{
  mapPoints.clear();
  mapPoints.resize(mvFeatsLeft.size(), nullptr);
  int nCreated = 0;
  for (std::size_t idx = 0; idx < mvFeatsLeft.size(); ++idx)
  {
    auto pMp = mvpMapPoints[idx];
    if (pMp && !pMp->isBad())
    {
      mapPoints[idx] = pMp;
      continue;
    }
    cv::Mat p3dC = VirtualFrame::unProject(idx);
    if (!p3dC.empty())
    {
      cv::Mat p3dW = mRwc * p3dC + mtwc;
      mapPoints[idx] = MapPoint::create(p3dW);
      ++nCreated;
    }
  }
  mvpMapPoints = mapPoints;
  return nCreated;
}

/**
 * @brief 获取共视关键帧
 *
 * @param th    输入的相连阈值
 * @return std::vector<KeyFrame::SharedPtr>
 */
std::vector<KeyFrame::SharedPtr> VirtualFrame::getConnectedKfs(int th)
{
  std::map<KeyFrame::SharedPtr, std::size_t> connectKfsWeight;
  std::vector<KeyFrame::SharedPtr> connectKfs;
  for (auto &pMp : mvpMapPoints)
  {
    if (!pMp || pMp->isBad())
    {
      continue;
    }
    for (auto &obs : pMp->getObservation())
    {
      auto kfPtr = obs.first.lock();
      if (kfPtr && !kfPtr->isBad())
      {
        ++connectKfsWeight[kfPtr];
      }
    }
  }
  int maxWeight = 0;
  KeyFrame::SharedPtr maxKF;
  for (const auto &item : connectKfsWeight)
  {
    if (item.second >= th)
      connectKfs.push_back(item.first);
    if (item.second > maxWeight)
    {
      maxWeight = item.second;
      maxKF = item.first;
    }
  }
  mpRefKF = maxKF;
  return connectKfs;
}

/**
 * @brief 获取VirtualFrame中的地图点
 *
 * @return std::vector<MapPointPtr> 输出的地图点
 */
std::vector<MapPoint::SharedPtr> VirtualFrame::getMapPoints()
{
  std::unique_lock<std::mutex> lock(mMutexMapPoints);
  return mvpMapPoints;
}

/**
 * @brief 逆投影到相机坐标系中（根据特征点位置和深度）
 *
 * @param idx 左图特征点索引
 * @return cv::Mat 输出的相机坐标系下的3D点
 */
cv::Mat VirtualFrame::unProject(std::size_t idx)
{
  const cv::KeyPoint &lKp = mvFeatsLeft[idx];
  const double &depth = mvDepths[idx];
  if (depth < 0)
    return cv::Mat();
  float x = (lKp.pt.x - Camera::mfCx) / Camera::mfFx;
  float y = (lKp.pt.y - Camera::mfCy) / Camera::mfFy;
  cv::Mat p3dC(3, 1, CV_32F);
  p3dC.at<float>(0) = depth * x;
  p3dC.at<float>(1) = depth * y;
  p3dC.at<float>(2) = depth;
  return p3dC;
}

/**
 * @brief 给定区域，寻找特征点
 *
 * @param kp        输入的关键点信息
 * @param radius    输入的初步搜索半径（会根据特征点的金字塔层级调整）
 * @param minNLevel 输入的最小金字塔层级（包含）
 * @param maxNLevel 输入的最大金字塔层级（包含）
 * @return std::vector<std::size_t> 输出符合要求的特征点索引
 */
std::vector<std::size_t> VirtualFrame::findFeaturesInArea(const cv::KeyPoint &kp, float radius, int minNLevel, int maxNLevel)
{
  std::vector<std::size_t> vFeatures;
  radius = radius * getScaledFactor2(kp.octave);
  int minX = std::max(0, cvRound(kp.pt.x - radius));
  int maxX = std::min((int)mfMaxU, cvRound(kp.pt.x + radius));
  int minY = std::max(0, cvRound(kp.pt.y - radius));
  int maxY = std::min((int)mfMaxV, cvRound(kp.pt.y + radius));
  int minColID = cvFloor((float)minX / mnGridWidth);
  int maxColID = cvFloor((float)maxX / mnGridWidth);
  int minRowID = cvFloor((float)minY / mnGridHeight);
  int maxRowID = cvFloor((float)maxY / mnGridHeight);
  for (std::size_t rowID = minRowID; rowID <= maxRowID; ++rowID)
  {
    for (std::size_t colID = minColID; colID <= maxColID; ++colID)
    {
      for (const auto &featID : mGrids[rowID][colID])
      {
        int octave = mvFeatsLeft[featID].octave;
        if (octave <= maxNLevel && octave >= minNLevel)
          vFeatures.push_back(featID);
      }
    }
  }
  return vFeatures;
}

/**
 * @brief 将世界坐标系下的3d点投影到像素坐标系下
 *
 * @param p3dW          输入的世界坐标系下3D点的坐标
 * @param isPositive    输出的投影到相机坐标系下，深度是否为正
 * @return cv::Point2f
 */
cv::Point2f VirtualFrame::project2UV(const cv::Mat &p3dW, bool &isPositive)
{
  cv::Mat p3dC;
  cv::Point2f point;
  {
    std::unique_lock<std::mutex> lock(mPoseMutex);
    p3dC = mRcw * p3dW + mtcw;
  }
  float z = p3dC.at<float>(2, 0);
  point.x = p3dC.at<float>(0, 0) / z * Camera::mfFx + Camera::mfCx;
  point.y = p3dC.at<float>(1, 0) / z * Camera::mfFy + Camera::mfCy;
  isPositive = z > 0;
  return point;
}

/**
 * @brief VirtualFrame的默认构造函数，供关键帧的流构造使用
 *
 */
VirtualFrame::VirtualFrame()
    : mbBowComputed(true)
    , mnID(0)
{
}

/**
 * @brief 参考关键帧，就是基于哪个关键帧做的位姿优化
 * @details
 *      1. 恒速模型跟踪：参考关键帧就是上一帧的参考关键帧
 *      2. 跟踪参考关键帧：跟踪线程最新创建的关键帧
 *      3. 跟踪局部地图：变成和自己共视程度最高的关键帧
 * @return KeyFrame::SharedPtr 返回参考关键帧
 */
KeyFrame::SharedPtr Frame::getRefKF() { return mpRefKF; }

std::vector<float> VirtualFrame::mvfScaledFactors;
bool VirtualFrame::mbScaled = false;
std::size_t VirtualFrame::mnNextID = 0;
unsigned VirtualFrame::mnGridHeight = 48;
unsigned VirtualFrame::mnGridWidth = 64;
} // namespace ORB_SLAM2_ROS2