#include "ORB_SLAM2/Camera.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 相机坐标系下的3d点投影到像素坐标系中
 * 
 * @param p3dC  输入的相机坐标系下的3D点
 * @param p2d   输出的像素坐标系下的2D点
 */
void Camera::project(const cv::Mat &p3dC, cv::Point2f &p2d){
    float x = p3dC.at<float>(0) / p3dC.at<float>(2);
    float y = p3dC.at<float>(1) / p3dC.at<float>(2);
    float u = mfFx * x + mfCx;
    float v = mfFy * y + mfCy;
    p2d.x = u;
    p2d.y = v;
}

/// 相机类型的静态变量
float Camera::mfBf = 386.1448;
float Camera::mfBl = 0.537166;
float Camera::mfFx = 718.856;
float Camera::mfFy = 718.856;
float Camera::mfCx = 607.1928;
float Camera::mfCy = 185.2157;
cv::Mat Camera::mK = (cv::Mat_<float>(3, 3) << mfFx, 0, mfCx, 0, mfFy, mfCy, 0, 0, 1);
cv::Mat Camera::mKInv = mK.inv();
} // namespace ORB_SLAM2_ROS2
