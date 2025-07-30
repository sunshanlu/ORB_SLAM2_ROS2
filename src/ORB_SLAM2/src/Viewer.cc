#include <unordered_set>

#include <pangolin/pangolin.h>

#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/LocalMapping.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/Tracking.h"
#include "ORB_SLAM2/Viewer.h"

namespace ORB_SLAM2_ROS2
{
using namespace std::chrono_literals;
using namespace pangolin;

Viewer::Viewer(MapPtr pMap, TrackingPtr pTracker)
    : mpMap(pMap)
    , mpTracker(pTracker)
    , mbIsStop(false)
    , mbReqestStop(false)
{
  mCurrPose = SE3(Mat3::Identity(), Vec3::Zero());
}

void Viewer::run()
{
  CreateWindowAndBind("ORB_SLAM2", 1080, 720);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  OpenGlRenderState sCamera(ProjectionMatrix(1080, 720, 420, 420, 540, 680, 0.1, 2000), ModelViewLookAt(0, -100, 0, 0, 0, 0, AxisZ));
  Handler3D handler(sCamera);
  View &d_cam = CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0).SetHandler(&handler);

  glPointSize(2);
  while (!ShouldQuit() && !isRequestStop())
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    d_cam.Activate(sCamera);

    bool updateMap = mpMap->getUpdate();
    if (updateMap)
    {
      mpMap->setUpdate(false);
      setAllMapPoints();
      setAllKeyFrames();
    }
    bool updateTracker = mpTracker.lock()->getUpdate();
    if (updateTracker)
    {
      mpTracker.lock()->setUpdate(false);
      setTrackingMps();
    }

    drawCurrentFrame();
    drawTrackingMapPoints(true);
    drawKeyFrames(true, true);
    drawAllMapPoints(true);

    FinishFrame();
    std::this_thread::sleep_for(7ms);
  }
  if (isRequestStop())
    stop();
}

/**
 * @brief 绘制位姿
 *
 * @param Twc 待绘制的位姿Twc
 */
void Viewer::drawPose(const SE3 &Twc)
{
  Vec3 center = Twc.translation();
  Vec3 p1 = Twc * Vec3(mfdx, mfdy, mfdz);
  Vec3 p2 = Twc * Vec3(mfdx, -mfdy, mfdz);
  Vec3 p3 = Twc * Vec3(-mfdx, -mfdy, mfdz);
  Vec3 p4 = Twc * Vec3(-mfdx, mfdy, mfdz);

  std::vector<Vec3> vPts{p1, p2, p3, p4, p1};
  for (std::size_t idx = 0; idx < 4; ++idx)
  {
    glVertex3f(vPts[idx][0], vPts[idx][1], vPts[idx][2]);
    glVertex3f(vPts[idx + 1][0], vPts[idx + 1][1], vPts[idx + 1][2]);

    glVertex3f(vPts[idx][0], vPts[idx][1], vPts[idx][2]);
    glVertex3f(center[0], center[1], center[2]);
  }
}

/**
 * @brief 绘制当前帧
 *
 */
void Viewer::drawCurrentFrame()
{
  {
    std::unique_lock<std::mutex> lock(mMutexCurrFrame);
    glLineWidth(2);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    drawPose(mCurrPose);
    glEnd();
    drawCurrImg();
  }
}

/**
 * @brief 绘制当前跟踪帧对应图像
 * @details
 *      1. 含有地图点的特征点
 *      2. 地图中的关键帧和地图点数目
 */
void Viewer::drawCurrImg()
{
  if (mCurrImg.empty())
    return;
  if (mCurrImg.channels() < 3)
    cv::cvtColor(mCurrImg, mCurrImg, cv::COLOR_GRAY2BGR);

  int nTracked = 0;
  auto &kps = mpFrame->getLeftKeyPoints();
  auto pMps = mpFrame->getMapPoints();
  for (std::size_t idx = 0; idx < kps.size(); ++idx)
  {
    auto pMp = pMps[idx];
    auto &kp = kps[idx];
    if (pMp && !pMp->isBad() && pMp->isInMap())
    {
      cv::Point2f p0(kp.pt.x - 5, kp.pt.y - 5);
      cv::Point2f p1(kp.pt.x + 5, kp.pt.y + 5);
      cv::rectangle(mCurrImg, p0, p1, cv::Scalar(0, 255, 0));
      cv::circle(mCurrImg, kp.pt, 2, cv::Scalar(0, 255, 0), -1);
      ++nTracked;
    }
  }
  std::stringstream s;
  if (mState == TrackingState::LOST)
    s << "Track Lost";
  else
  {
    int nKfs = mpMap->keyFramesInMap();
    int nMps = mpMap->mapPointsInMap();
    s << "KFs: " << nKfs << ", MPs: " << nMps << ", Matches: " << nTracked;
  }
  int baseline = 0;
  cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);
  cv::Mat imText = cv::Mat::zeros(mCurrImg.rows + textSize.height + 10, mCurrImg.cols, mCurrImg.type());
  mCurrImg.copyTo(imText.rowRange(0, mCurrImg.rows).colRange(0, mCurrImg.cols));
  cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
  cv::imshow("跟踪线程: 当前帧", imText);
  cv::waitKey(3);
}

/**
 * @brief 绘制地图点
 *
 */
void Viewer::drawMapPoints(const std::vector<MapPointPtr> &vMps)
{
  for (const auto &pMp : vMps)
  {
    if (!pMp || pMp->isBad())
      continue;
    auto p = pMp->getPos();
    glVertex3f(p.at<float>(0), p.at<float>(1), p.at<float>(2));
  }
}

/**
 * @brief 绘制关键帧
 *
 */
void Viewer::drawKeyFrames(const bool &bDrawKF, const bool &bDrawGraph)
{
  glBegin(GL_LINES);
  if (bDrawKF)
  {
    glLineWidth(2);
    glColor3f(0.0f, 0.0f, 1.0f);
    for (const auto &kf : mvAllKeyFrames)
    {
      if (kf && !kf->isBad())
        drawPose(convertMat2SE3(kf->getPoseInv()));
    }
  }
  if (bDrawGraph)
  {
    glLineWidth(1);
    glColor3f(0.0f, 0.0f, 1.0f);
    for (const auto &kf : mvAllKeyFrames)
    {
      if (!kf || kf->isBad())
        continue;
      drawGraph(kf);
    }
  }
  glEnd();
}

/// 绘制所有地图点
void Viewer::drawAllMapPoints(const bool &bDrawMP)
{
  if (bDrawMP)
  {
    glColor3f(0.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    drawMapPoints(mvAllMapPoints);
    glEnd();
  }
}

/// 绘制跟踪线程地图点
void Viewer::drawTrackingMapPoints(const bool &bDrawMP)
{
  if (bDrawMP)
  {
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    drawMapPoints(mvTrackingMps);
    glEnd();
  }
}

/// 将cv::Mat转换为SE3类型来表示位姿
Viewer::SE3 Viewer::convertMat2SE3(const cv::Mat &Twc)
{
  Mat3 Rwc;
  Vec3 twc;
  Rwc << Twc.at<float>(0, 0), Twc.at<float>(0, 1), Twc.at<float>(0, 2), Twc.at<float>(1, 0), Twc.at<float>(1, 1), Twc.at<float>(1, 2), Twc.at<float>(2, 0),
      Twc.at<float>(2, 1), Twc.at<float>(2, 2);
  twc << Twc.at<float>(0, 3), Twc.at<float>(1, 3), Twc.at<float>(2, 3);
  return SE3(Rwc, twc);
}

/**
 * @brief 设置当前关键帧
 * @details
 *      1. 设置当前关键的位姿
 *      2. 设置跟踪线程的图像
 *      3. 设置跟踪线程的普通帧指针
 * @param Twc           输入的当前帧位姿
 * @param trackImage    输入的当前帧图像
 * @param pCurrFrame    输入的当前帧指针
 */
void Viewer::setCurrFrame(cv::Mat trackImage, FramePtr pCurrFrame, TrackingState state)
{
  std::unique_lock<std::mutex> lock(mMutexCurrFrame);
  if (state == TrackingState::OK)
  {
    mCurrPose = convertMat2SE3(pCurrFrame->getPoseInv());
    mCurrImg = trackImage;
    mpFrame = pCurrFrame;
    mState = state;
    return;
  }
  mState = state;
}

/// 设置跟踪线程地图点
void Viewer::setTrackingMps()
{
  mvTrackingMps.clear();
  mvTrackingMps = mpTracker.lock()->getLocalMps();
}

/// 设置所有关键帧位姿
void Viewer::setAllKeyFrames()
{
  mvAllKeyFrames.clear();
  mvAllKeyFrames = mpMap->getAllKeyFrames();
}

/// 设置所有地图点
void Viewer::setAllMapPoints()
{
  mvAllMapPoints.clear();
  mvAllMapPoints = mpMap->getAllMapPoints();
}

/// 绘制graph
void Viewer::drawGraph(const KeyFramePtr &pKf)
{
  std::vector<KeyFramePtr> vKfs = pKf->getConnectedKfs(100);
  std::unordered_set<KeyFramePtr> sKfSet(vKfs.begin(), vKfs.end());
  auto pParent = pKf->getParent().lock();
  if (pParent && !pParent->isBad())
    sKfSet.insert(pParent);
  auto vLoopEdges = pKf->getLoopEdges();
  for (auto &pLoop : vLoopEdges)
    sKfSet.insert(pLoop);
  cv::Mat pos0 = pKf->getFrameCenter();
  for (auto &pkf2 : sKfSet)
  {
    cv::Mat pos1 = pkf2->getFrameCenter();
    glVertex3f(pos0.at<float>(0), pos0.at<float>(1), pos0.at<float>(2));
    glVertex3f(pos1.at<float>(0), pos1.at<float>(1), pos1.at<float>(2));
  }
}

/// Viewer的静态变量
float Viewer::mfdx = 0.f;
float Viewer::mfdy = 0.f;
float Viewer::mfdz = 0.f;

} // namespace ORB_SLAM2_ROS2
