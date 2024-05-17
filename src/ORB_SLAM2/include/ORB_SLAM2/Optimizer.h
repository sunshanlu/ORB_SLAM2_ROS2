#pragma once
#include <memory>

#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sim3/sim3.h>
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_ROS2 {
class Frame;
class KeyFrame;
class MapPoint;
class Sim3Ret;
class Map;

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

    /// 将SIM3Ret类型转换为g2o::Sim3类型
    static g2o::Sim3 ConvertSim3G2o(const Sim3Ret &Scm);

    /// 将g2o::Sim3类型转换为SIM3Ret类型
    static Sim3Ret ConvertG2o2Sim3(const g2o::Sim3 &Scm);
};

class Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef g2o::BlockSolver_6_3 BSSE3;
    typedef g2o::BlockSolver_7_3 BSSIM3;
    typedef g2o::LinearSolverDense<BSSE3::PoseMatrixType> LSSE3Dense;
    typedef g2o::LinearSolverDense<BSSIM3::PoseMatrixType> LSSim3Dense;
    typedef g2o::LinearSolverEigen<BSSIM3::PoseMatrixType> LSSim3Eigen;
    typedef g2o::LinearSolverEigen<BSSE3::PoseMatrixType> LSSE3Eigen;
    typedef std::shared_ptr<Frame> FramePtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<MapPoint> MapPointPtr;
    typedef std::shared_ptr<Map> MapPtr;
    typedef g2o::OptimizableGraph::Edge *EdgeDBKey;
    typedef std::tuple<MapPointPtr, KeyFramePtr, std::size_t, bool> EdgeDBVal;
    typedef std::unordered_map<EdgeDBKey, EdgeDBVal> EdgeDB;
    typedef std::vector<cv::DMatch> Matches;
    typedef std::map<KeyFramePtr, std::set<KeyFramePtr>> LoopConnection;
    typedef std::map<KeyFramePtr, Sim3Ret> KeyFrameAndSim3;

    Optimizer() = default;

    /// 仅优化位姿
    static int OptimizePoseOnly(FramePtr pFrame);

    /// 局部建图线程的局部地图优化
    static void OptimizeLocalMap(KeyFramePtr pkframe, bool &isStop);

    /// 计算Sim3变换矩阵
    static int OptimizeSim3(KeyFramePtr pCurr, KeyFramePtr pMatch, Matches &inLier, Sim3Ret &g2oScm,
                            bool bFixedScale = true);

    /// 优化本质图
    static void optimizeEssentialGraph(const LoopConnection &mLoopConnections, MapPtr mpMap, KeyFramePtr pLoopKf,
                                       KeyFramePtr pCurrKf, const int &graphTh, const KeyFrameAndSim3 &mCorrectedG2oScw,
                                       const KeyFrameAndSim3 &mNoCorrectedG2oScw, bool bFixedScale = true);

    /// 计算3/4分位数
    static double ComputeThirdQuartile(const std::vector<double> &data);

private:
    static float deltaMono;   ///< 单目二自由度
    static float deltaStereo; ///< 双目三自由度
    static float deltaSim3;   ///< SIM3二自由度
};
} // namespace ORB_SLAM2_ROS2