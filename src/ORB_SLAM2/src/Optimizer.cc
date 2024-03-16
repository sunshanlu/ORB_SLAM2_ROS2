#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/edge_project_stereo_xyz_onlypose.h>
#include <g2o/types/sba/edge_project_xyz_onlypose.h>
#include <g2o/types/sba/vertex_se3_expmap.h>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/Optimizer.h"

namespace ORB_SLAM2_ROS2 {
typedef g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType> LinearSolverType;
typedef g2o::BlockSolver_6_3 BlockSolverType;

/**
 * @brief 跟踪线程部分的优化器，仅优化帧位姿
 * @details
 *      1. 使用单目误差边和双目误差边两种方式进行优化
 *      2. 优化过程中，设置缩放因子倒数的平方作为信息矩阵（金字塔层级越高，比重越小）
 *      3. 在优化过程之前，需要设置帧的位姿
 *      4. 优化过程中会将明显的外点进行剔除（误差较大的边对应的点 + 投影超出范围的地图点）
 * @param pFrame    输入的待优化位姿的帧
 * @return int      输出优化内点的个数
 */
int Optimizer::OptimizePoseOnly(Frame::SharedPtr pFrame) {
    auto lm = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer graph;
    graph.setAlgorithm(lm);

    auto poseVertex = new g2o::VertexSE3Expmap();
    poseVertex->setId(0);
    graph.addVertex(poseVertex);
    const float deltaMono = std::sqrt(5.991);   ///< 单目二自由度
    const float deltaStereo = std::sqrt(7.815); ///< 双目三自由度

    std::map<std::size_t, g2o::EdgeSE3ProjectXYZOnlyPose *> monoEdges;
    std::map<std::size_t, g2o::EdgeStereoSE3ProjectXYZOnlyPose *> stereoEdges;
    std::vector<bool> inLier(pFrame->mvFeatsLeft.size(), true);

    /// 添加位姿节点和误差边
    int edges = 0;
    std::vector<cv::Mat> mapPointPoses;
    for (std::size_t idx = 0; idx < pFrame->mvFeatsLeft.size(); ++idx) {
        auto &pMp = pFrame->mvpMapPoints[idx];
        const auto &rightU = pFrame->mvFeatsRightU[idx];
        const auto &fKp = pFrame->mvFeatsLeft[idx];
        cv::Mat pos;
        if (pMp && !pMp->isBad()) {
            pos = pMp->getPos();
            auto rk = new g2o::RobustKernelHuber();
            if (rightU < 0) {
                rk->setDelta(deltaMono);
                auto edgeMono = new g2o::EdgeSE3ProjectXYZOnlyPose();
                edgeMono->setVertex(0, poseVertex);
                edgeMono->fx = Camera::mfFx;
                edgeMono->fy = Camera::mfFy;
                edgeMono->cx = Camera::mfCx;
                edgeMono->cy = Camera::mfCy;
                edgeMono->Xw << (double)pos.at<float>(0), (double)pos.at<float>(1), (double)pos.at<float>(2);
                edgeMono->setMeasurement(g2o::Vector2((double)fKp.pt.x, (double)fKp.pt.y));
                edgeMono->setInformation(Eigen::Matrix2d::Identity() * pFrame->getScaledFactorInv2(fKp.octave));
                edgeMono->setRobustKernel(rk);
                monoEdges.insert(std::make_pair(idx, edgeMono));
                graph.addEdge(edgeMono);
            } else {
                rk->setDelta(deltaStereo);
                auto edgeStereo = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                edgeStereo->setVertex(0, poseVertex);
                edgeStereo->fx = Camera::mfFx;
                edgeStereo->fy = Camera::mfFy;
                edgeStereo->cx = Camera::mfCx;
                edgeStereo->cy = Camera::mfCy;
                edgeStereo->bf = Camera::mfBf;
                edgeStereo->Xw << (double)pos.at<float>(0), (double)pos.at<float>(1), (double)pos.at<float>(2);
                edgeStereo->setMeasurement(g2o::Vector3((double)fKp.pt.x, (double)fKp.pt.y, (double)rightU));
                edgeStereo->setInformation(Eigen::Matrix3d::Identity() * pFrame->getScaledFactorInv2(fKp.octave));
                edgeStereo->setRobustKernel(rk);
                stereoEdges.insert(std::make_pair(idx, edgeStereo));
                graph.addEdge(edgeStereo);
            }
            ++edges;
        } else {
            pMp = nullptr;
            inLier[idx] = false;
        }
        mapPointPoses.push_back(pos);
    }

    auto se3 = Converter::ConvertTcw2SE3(pFrame->mRcw, pFrame->mtcw);
    int nBad = 0;
    for (int i = 0; i < 4; ++i) {
        nBad = 0;
        poseVertex->setEstimate(se3);
        graph.initializeOptimization(0);
        graph.optimize(10);

        /// 寻找超出要求的误差边
        for (auto &item : monoEdges) {
            auto &edge = item.second;
            const auto &octave = pFrame->mvFeatsLeft[item.first].octave;
            float sigma2 = pFrame->getScaledFactor2(octave);
            if (!inLier[item.first]) {
                edge->computeError();
            }
            auto error = edge->chi2();
            if (edge->chi2() > 5.991 * sigma2) {
                inLier[item.first] = false;
                edge->setLevel(1);
                ++nBad;
            } else {
                inLier[item.first] = true;
                edge->setLevel(0);
            }
        }
        for (auto &item : stereoEdges) {
            auto &edge = item.second;
            const auto &octave = pFrame->mvFeatsLeft[item.first].octave;
            float sigma2 = pFrame->getScaledFactor2(octave);
            if (!inLier[item.first]) {
                edge->computeError();
            }
            auto error = edge->chi2();
            if (edge->chi2() > 7.815 * sigma2) {
                inLier[item.first] = false;
                edge->setLevel(1);
                ++nBad;
            } else {
                inLier[item.first] = true;
                edge->setLevel(0);
            }
        }
    }
    for (std::size_t idx = 0; idx < inLier.size(); ++idx) {
        if (!inLier[idx])
            continue;
        bool isPositive = false;
        auto pointUV = pFrame->project2UV(mapPointPoses[idx], isPositive);
        if (!isPositive || pointUV.x > pFrame->mnMaxU || pointUV.x < 0 || pointUV.y > pFrame->mnMaxV || pointUV.y < 0) {
            inLier[idx] = false;
            ++nBad;
        }
    }
    for (std::size_t idx = 0; idx < inLier.size(); ++idx) {
        if (!inLier[idx])
            pFrame->mvpMapPoints[idx] = nullptr;
    }
    auto se3Optimized = poseVertex->estimate();
    pFrame->setPose(Converter::ConvertSE32Tcw(se3Optimized));
    return edges - nBad;
}

/**
 * @brief 将opencv的Rcw和tcw转换为g2o的SE3
 *
 * @param RcwCV 输入的Rcw
 * @param tcwCV 输入的tcw
 * @return g2o::SE3Quat 输出的SE3
 */
g2o::SE3Quat Converter::ConvertTcw2SE3(const cv::Mat &RcwCV, const cv::Mat &tcwCV) {
    Eigen::Matrix3d RcwEigen;
    Eigen::Vector3d tcwEigen;
    RcwEigen << (double)RcwCV.at<float>(0, 0), (double)RcwCV.at<float>(0, 1), (double)RcwCV.at<float>(0, 2),
        (double)RcwCV.at<float>(1, 0), (double)RcwCV.at<float>(1, 1), (double)RcwCV.at<float>(1, 2),
        (double)RcwCV.at<float>(2, 0), (double)RcwCV.at<float>(2, 1), (double)RcwCV.at<float>(2, 2);
    tcwEigen.x() = (double)tcwCV.at<float>(0, 0);
    tcwEigen.y() = (double)tcwCV.at<float>(1, 0);
    tcwEigen.z() = (double)tcwCV.at<float>(2, 0);
    Eigen::Quaterniond qcwEigen(RcwEigen);
    qcwEigen.normalize();
    return g2o::SE3Quat(qcwEigen, tcwEigen);
}

/**
 * @brief 将g2o类型的位姿矩阵转换为OpenCV类型表示的位姿矩阵
 *
 * @param SE3   输入的g2o类型的位姿矩阵
 * @return cv::Mat 输出的OpenCV类型的位姿矩阵
 */
cv::Mat Converter::ConvertSE32Tcw(const g2o::SE3Quat &SE3) {
    cv::Mat Tcw(4, 4, CV_32F);
    Tcw.at<float>(0, 0) = (float)SE3.rotation().matrix()(0, 0);
    Tcw.at<float>(0, 1) = (float)SE3.rotation().matrix()(0, 1);
    Tcw.at<float>(0, 2) = (float)SE3.rotation().matrix()(0, 2);
    Tcw.at<float>(1, 0) = (float)SE3.rotation().matrix()(1, 0);
    Tcw.at<float>(1, 1) = (float)SE3.rotation().matrix()(1, 1);
    Tcw.at<float>(1, 2) = (float)SE3.rotation().matrix()(1, 2);
    Tcw.at<float>(2, 0) = (float)SE3.rotation().matrix()(2, 0);
    Tcw.at<float>(2, 1) = (float)SE3.rotation().matrix()(2, 1);
    Tcw.at<float>(2, 2) = (float)SE3.rotation().matrix()(2, 2);

    Tcw.at<float>(0, 3) = (float)SE3.translation().x();
    Tcw.at<float>(1, 3) = (float)SE3.translation().y();
    Tcw.at<float>(2, 3) = (float)SE3.translation().z();

    Tcw.at<float>(3, 0) = 0.0f;
    Tcw.at<float>(3, 1) = 0.0f;
    Tcw.at<float>(3, 2) = 0.0f;
    Tcw.at<float>(3, 3) = 1.0f;
    return Tcw;
}
} // namespace ORB_SLAM2_ROS2