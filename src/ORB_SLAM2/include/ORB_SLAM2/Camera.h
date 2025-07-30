#pragma once

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_ROS2
{

enum class CameraType
{
  Stereo,
  RGBD
};

struct Camera
{

  /// 相机坐标系下的3d点投影到像素坐标系中
  static void project(const cv::Mat &p3dC, cv::Point2f &p2d);

  /// 离散关键点的畸变去除
  static void undistortPoints(std::vector<cv::KeyPoint> &keyPoints);

  static float mfBf;         ///< 相机基线 * 焦距
  static float mfBl;         ///< 相机基线
  static float mfFx;         ///< 相机焦距fx
  static float mfFy;         ///< 相机焦距fy
  static float mfCx;         ///< 相机参数cx
  static float mfCy;         ///< 相机参数cy
  static cv::Mat mK;         ///< 相机内参矩阵
  static cv::Mat mKInv;      ///< 相机内参矩阵
  static cv::Mat mDistCoeff; ///< 相机的畸变系数矩阵
  static CameraType mType;   ///< 相机类型
};
} // namespace ORB_SLAM2_ROS2