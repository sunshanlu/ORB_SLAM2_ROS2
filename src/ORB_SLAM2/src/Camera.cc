#include "ORB_SLAM2/Camera.h"

namespace ORB_SLAM2_ROS2 {
float Camera::mfBf = 386.1448;
float Camera::mfBl = 0.537166;
float Camera::mfFx = 718.856;
float Camera::mfFy = 718.856;
float Camera::mfCx = 607.1928;
float Camera::mfCy = 185.2157;
cv::Mat Camera::mK = (cv::Mat_<float>(3, 3) << mfFx, 0, mfCx, 0, mfFy, mfCy, 0, 0, 1);
cv::Mat Camera::mKInv = mK.inv();
} // namespace ORB_SLAM2_ROS2
