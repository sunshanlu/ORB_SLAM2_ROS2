#include <opencv2/imgproc.hpp>

#include "ORB_SLAM2/Camera.h"

namespace ORB_SLAM2_ROS2
{

/**
 * @brief 相机坐标系下的3d点投影到像素坐标系中
 *
 * @param p3dC  输入的相机坐标系下的3D点
 * @param p2d   输出的像素坐标系下的2D点
 */
void Camera::project(const cv::Mat &p3dC, cv::Point2f &p2d)
{
  float x = p3dC.at<float>(0) / p3dC.at<float>(2);
  float y = p3dC.at<float>(1) / p3dC.at<float>(2);
  float u = mfFx * x + mfCx;
  float v = mfFy * y + mfCy;
  p2d.x = u;
  p2d.y = v;
}

/**
 * @brief 相机的去畸变模型
 *
 * @param keyPoints 输入输出的2d关键点
 */
void Camera::undistortPoints(std::vector<cv::KeyPoint> &keyPoints)
{
  if (mDistCoeff.empty() || !mDistCoeff.at<float>(0) || keyPoints.empty())
    return;

  std::vector<cv::Point2f> points;
  std::for_each(keyPoints.begin(), keyPoints.end(), [&](const cv::KeyPoint &kp) { points.push_back(kp.pt); });
  cv::undistortPoints(points, points, mK, mDistCoeff, cv::noArray(), mK);
  for (std::size_t idx = 0; idx < keyPoints.size(); ++idx)
    keyPoints[idx].pt = points[idx];
}

/// 相机类型的静态变量
float Camera::mfBf = 0;
float Camera::mfBl = 0;
float Camera::mfFx = 0;
float Camera::mfFy = 0;
float Camera::mfCx = 0;
float Camera::mfCy = 0;
CameraType Camera::mType = CameraType::Stereo;
cv::Mat Camera::mK;
cv::Mat Camera::mKInv;
cv::Mat Camera::mDistCoeff;
} // namespace ORB_SLAM2_ROS2
