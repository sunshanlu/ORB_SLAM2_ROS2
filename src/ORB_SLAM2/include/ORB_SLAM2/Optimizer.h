#pragma once
#include <memory>

#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_ROS2 {
class Frame;
class KeyFrame;
class MapPoint;

class Converter {
public:
    /// 将cv::Mat类型的Rcw和tcw表示的SE3转换成g2o::SE3Quat
    static g2o::SE3Quat ConvertTcw2SE3(const cv::Mat &Rcw, const cv::Mat &tcw);

    /// 将g2o::SE3Quat转换成cv::Mat类型表示位姿
    static cv::Mat ConvertSE32Tcw(const g2o::SE3Quat &SE3);

    /// 将cv::Mat类型的Pw转换为g2o::Vector3类型
    static g2o::Vector3 ConvertPw2Vector3(const cv::Mat &Pw);

    /// 将g2o::Vector3类型的Pw转换为cv::Mat类型
    static cv::Mat ConvertVector32Pw(const g2o::Vector3 &Pw);
};

class Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    typedef std::shared_ptr<Frame> FramePtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<MapPoint> MapPointPtr;
    typedef g2o::OptimizableGraph::Edge *EdgeDBKey;
    typedef std::tuple<MapPointPtr, KeyFramePtr, std::size_t, bool> EdgeDBVal;
    typedef std::unordered_map<EdgeDBKey, EdgeDBVal> EdgeDB;

    Optimizer() = default;

    /// 仅优化位姿
    static int OptimizePoseOnly(FramePtr pFrame);

    /// 局部建图线程的局部地图优化
    static void OptimizeLocalMap(KeyFramePtr pkframe, bool &isStop);

    /// 计算3/4分位数
    static double ComputeThirdQuartile(const std::vector<double> &data);

private:
    static float deltaMono;   ///< 单目二自由度
    static float deltaStereo; ///< 双目三自由度
};
} // namespace ORB_SLAM2_ROS2