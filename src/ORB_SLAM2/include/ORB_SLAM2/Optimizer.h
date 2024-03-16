#pragma once
#include <memory>

#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_ROS2 {
class Frame;

class Converter {
public:
    /// 将cv::Mat类型的Rcw和tcw表示的SE3转换成g2o::SE3Quat
    static g2o::SE3Quat ConvertTcw2SE3(const cv::Mat &Rcw, const cv::Mat &tcw);

    /// 将g2o::SE3Quat转换成cv::Mat类型表示位姿
    static cv::Mat ConvertSE32Tcw(const g2o::SE3Quat &SE3);
};

class Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> FramePtr;

    Optimizer() = default;
    static int OptimizePoseOnly(FramePtr pFrame);
};
} // namespace ORB_SLAM2_ROS2