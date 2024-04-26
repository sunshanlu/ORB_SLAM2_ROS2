#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/Optimizer.h"

namespace ORB_SLAM2_ROS2 {

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

    std::map<std::size_t, g2o::EdgeSE3ProjectXYZOnlyPose *> monoEdges;
    std::map<std::size_t, g2o::EdgeStereoSE3ProjectXYZOnlyPose *> stereoEdges;
    std::vector<bool> inLier(pFrame->mvFeatsLeft.size(), true);

    /// 添加位姿节点和误差边
    int edges = 0;
    std::vector<cv::Mat> mapPointPoses;
    auto mapPoints = pFrame->getMapPoints();
    auto &kps = pFrame->getLeftKeyPoints();
    std::vector<double> monoEdgeErrors;
    std::vector<double> stereoEdgeErrors;
    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> monoEdgesVec;
    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> stereoEdgesVec;

    for (std::size_t idx = 0; idx < mapPoints.size(); ++idx) {
        auto &pMp = mapPoints[idx];
        const auto &rightU = pFrame->getRightU(idx);
        const auto &fKp = kps[idx];
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
                edgeMono->computeError();
                double error = edgeMono->chi2();
                monoEdgeErrors.push_back(error);
                monoEdgesVec.push_back(edgeMono);
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
                edgeStereo->computeError();
                double error = edgeStereo->chi2();
                stereoEdgeErrors.push_back(error);
                stereoEdgesVec.push_back(edgeStereo);
            }
            ++edges;
        } else {
            pMp = nullptr;
            inLier[idx] = false;
        }
        mapPointPoses.push_back(pos);
    }

    // 为了防止大误差边带离收敛范围，需要采用3/4中位数和最大值匹配的方式进行部分边的固定
    // std::size_t monoN = monoEdgeErrors.size();
    // std::size_t stereoN = stereoEdgeErrors.size();
    // if (monoN > 0) {
    //     auto errorsCp = monoEdgeErrors;
    //     std::sort(errorsCp.begin(), errorsCp.end(), std::less<double>());
    //     double thirdQuartile = ComputeThirdQuartile(errorsCp);
    //     if (errorsCp[monoN - 1] / thirdQuartile > 10) {
    //         for (std::size_t idx = 0; idx < monoN; ++idx) {
    //             auto &edge = monoEdgesVec[idx];
    //             if (monoEdgeErrors[idx] > 5 * thirdQuartile)
    //                 edge->setLevel(1);
    //         }
    //     }
    // }
    // if (stereoN > 0) {
    //     auto errorsCp = stereoEdgeErrors;
    //     std::sort(errorsCp.begin(), errorsCp.end(), std::less<double>());
    //     double thirdQuartile = ComputeThirdQuartile(errorsCp);
    //     if (errorsCp[stereoN - 1] / thirdQuartile > 10) {
    //         for (std::size_t idx = 0; idx < stereoN; ++idx) {
    //             auto &edge = stereoEdgesVec[idx];
    //             if (stereoEdgeErrors[idx] > 5 * thirdQuartile)
    //                 edge->setLevel(1);
    //         }
    //     }
    // }

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
            if (i == 2)
                edge->setRobustKernel(nullptr);
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
            if (i == 2)
                edge->setRobustKernel(nullptr);
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
        else {
            pFrame->mvpMapPoints[idx]->addInlierInTrack();
        }
    }
    auto se3Optimized = poseVertex->estimate();
    pFrame->setPose(Converter::ConvertSE32Tcw(se3Optimized));
    return edges - nBad;
}

/**
 * @brief 局部建图线程中的局部地图优化
 * @details
 *      1. 找到当前关键帧的一阶相连关键帧
 *      2. 使用地图点的方式找到二阶相连关键帧
 *      3. 将二阶相连关键帧对应的顶点固定（也包括id为0的关键帧进行固定）
 *      4. 进行两次优化
 *          (1) 第一次优化5次
 *          (2) 第二次优化10次
 *          (3) 在每次优化后，将误差边较大的部分不参与下次优化
 *      5. 对误差较大的边对应的地图点进行观测剔除
 *          (1) 关键帧：对应的地图点位置设置为nullptr
 *          (2) 地图点：对应的Observe进行删除
 *      6. 设置关键帧位姿和地图点位置
 * @param pkframe   输入的局部建图线程的当前关键帧
 * @param isStop    是否终止BA（当跟踪线程插入关键帧的时候，为true）
 *      1. g2o会接受一个bool类型的指针，在每一次迭代开始前进行判断是否要开始优化
 *      2. 在第一阶段的优化前，判断是否需要停止BA，如果停止，直接return
 *      3. 在第二阶段的优化前，判断是否需要停止BA，如果停止，直接跳过第二次优化，直接进行下一步操作
 */
void Optimizer::OptimizeLocalMap(KeyFramePtr pkframe, bool &isStop) {
    /// step1: 获取一阶相连和二阶相连误差边统计
    std::map<MapPointPtr, std::map<KeyFramePtr, std::size_t>> noFixedConnected;
    std::map<MapPointPtr, std::map<KeyFramePtr, std::size_t>> fixedConnected;
    auto connectedKFs = pkframe->getAllConnected();
    connectedKFs.insert({pkframe, 0});
    for (auto &connect : connectedKFs) {
        KeyFrame::SharedPtr pKf = connect.first.lock();
        if (!pKf || pKf->isBad())
            continue;
        auto mapPoints = pKf->getMapPoints();
        for (auto &pMp : mapPoints) {
            if (!pMp || pMp->isBad()) {
                continue;
            }
            auto obss = pMp->getObservation();
            std::map<KeyFramePtr, std::size_t> &noFixedConnectedi = noFixedConnected[pMp];
            std::map<KeyFramePtr, std::size_t> &fixedConnectedi = fixedConnected[pMp];
            for (auto &obs : obss) {
                auto pkf = obs.first.lock();
                if (!pkf || pkf->isBad())
                    continue;
                if (connectedKFs.find(pkf) != connectedKFs.end())
                    noFixedConnectedi.insert({pkf, obs.second});
                else
                    fixedConnectedi.insert({pkf, obs.second});
            }
        }
    }

    /// step2: 构造优化器
    auto lm = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer graph;
    graph.setAlgorithm(lm);
    graph.setForceStopFlag(&isStop);

    /// step3: 添加顶点信息和边信息
    std::size_t vertexID = 0, edgeID = 0;
    EdgeDB edgeDB;
    std::unordered_map<MapPointPtr, g2o::VertexPointXYZ *> landmarks;
    std::unordered_map<KeyFramePtr, g2o::VertexSE3Expmap *> frames;
    for (auto &noFixed : noFixedConnected) {
        auto pMp = noFixed.first;
        g2o::VertexPointXYZ *landmark;
        if (landmarks.find(pMp) == landmarks.end()) {
            cv::Mat position = pMp->getPos();
            landmark = new g2o::VertexPointXYZ();
            landmark->setId(vertexID++);
            auto v = Converter::ConvertPw2Vector3(position);
            landmark->setEstimate(v);
            landmark->setMarginalized(true);
            landmarks.insert({pMp, landmark});
            graph.addVertex(landmark);
        } else
            landmark = landmarks.at(pMp);
        auto &kfAndKps = noFixed.second;
        for (auto &item : kfAndKps) {
            auto pkf = item.first;
            g2o::VertexSE3Expmap *frame;
            if (frames.find(pkf) == frames.end()) {
                cv::Mat Rcw, tcw;
                pkf->getPose(Rcw, tcw);
                frame = new g2o::VertexSE3Expmap();
                frame->setId(vertexID++);
                frame->setEstimate(Converter::ConvertTcw2SE3(Rcw, tcw));
                if (pkf->getID() == 0)
                    frame->setFixed(true);
                frames.insert({pkf, frame});
                graph.addVertex(frame);
            } else
                frame = frames.at(pkf);
            const auto &kp = pkf->getLeftKeyPoint(item.second);
            const double &rightU = pkf->getRightU(item.second);
            auto rk = new g2o::RobustKernelHuber();
            if (rightU < 0) {
                rk->setDelta(deltaMono);
                auto edgeMono = new g2o::EdgeSE3ProjectXYZ();
                edgeMono->setId(edgeID++);
                edgeMono->setVertex(0, landmark);
                edgeMono->setVertex(1, frame);
                edgeMono->fx = Camera::mfFx;
                edgeMono->fy = Camera::mfFy;
                edgeMono->cx = Camera::mfCx;
                edgeMono->cy = Camera::mfCy;
                auto e = g2o::Vector2((double)kp.pt.x, (double)kp.pt.y);
                edgeMono->setMeasurement(e);
                /// 信息矩阵是协方差矩阵的逆
                edgeMono->setInformation(Eigen::Matrix2d::Identity() * pkf->getScaledFactorInv2(kp.octave));
                edgeMono->setRobustKernel(rk);
                graph.addEdge(edgeMono);
                edgeDB[edgeMono] = std::make_tuple(pMp, pkf, item.second, true);
            } else {
                rk->setDelta(deltaStereo);
                auto edgeStereo = new g2o::EdgeStereoSE3ProjectXYZ();
                edgeStereo->setId(edgeID++);
                edgeStereo->setVertex(0, landmark);
                edgeStereo->setVertex(1, frame);
                edgeStereo->fx = Camera::mfFx;
                edgeStereo->fy = Camera::mfFy;
                edgeStereo->cx = Camera::mfCx;
                edgeStereo->cy = Camera::mfCy;
                edgeStereo->bf = Camera::mfBf;
                auto e = g2o::Vector3((double)kp.pt.x, (double)kp.pt.y, (double)rightU);
                edgeStereo->setMeasurement(e);
                edgeStereo->setInformation(Eigen::Matrix3d::Identity() * pkf->getScaledFactorInv2(kp.octave));
                edgeStereo->setRobustKernel(rk);
                graph.addEdge(edgeStereo);
                edgeDB[edgeStereo] = std::make_tuple(pMp, pkf, item.second, false);
            }
        }
    }
    for (auto &fixed : fixedConnected) {
        auto pMp = fixed.first;
        g2o::VertexPointXYZ *landmark;
        if (landmarks.find(pMp) == landmarks.end()) {
            cv::Mat position = pMp->getPos();
            landmark = new g2o::VertexPointXYZ();
            landmark->setId(vertexID++);
            landmark->setEstimate(Converter::ConvertPw2Vector3(position));
            landmark->setMarginalized(true);
            landmarks.insert({pMp, landmark});
            graph.addVertex(landmark);
        } else
            landmark = landmarks.at(pMp);
        auto &kfAndKps = fixed.second;
        for (auto &item : kfAndKps) {
            auto pkf = item.first;
            g2o::VertexSE3Expmap *frame;
            if (frames.find(pkf) == frames.end()) {
                cv::Mat Rcw, tcw;
                pkf->getPose(Rcw, tcw);
                frame = new g2o::VertexSE3Expmap();
                frame->setId(vertexID++);
                frame->setEstimate(Converter::ConvertTcw2SE3(Rcw, tcw));
                frame->setFixed(true);
                frames.insert({pkf, frame});
                graph.addVertex(frame);
            } else
                frame = frames.at(pkf);
            const auto &kp = pkf->getLeftKeyPoint(item.second);
            const double &rightU = pkf->getRightU(item.second);
            auto rk = new g2o::RobustKernelHuber();
            if (rightU < 0) {
                rk->setDelta(deltaMono);
                auto edgeMono = new g2o::EdgeSE3ProjectXYZ();
                edgeMono->setId(edgeID++);
                edgeMono->setVertex(0, landmark);
                edgeMono->setVertex(1, frame);
                edgeMono->fx = Camera::mfFx;
                edgeMono->fy = Camera::mfFy;
                edgeMono->cx = Camera::mfCx;
                edgeMono->cy = Camera::mfCy;
                edgeMono->setMeasurement(g2o::Vector2((double)kp.pt.x, (double)kp.pt.y));
                /// 信息矩阵是协方差矩阵的逆
                edgeMono->setInformation(Eigen::Matrix2d::Identity() * pkf->getScaledFactorInv2(kp.octave));
                edgeMono->setRobustKernel(rk);
                graph.addEdge(edgeMono);
                edgeDB[edgeMono] = std::make_tuple(pMp, pkf, item.second, true);
            } else {
                rk->setDelta(deltaStereo);
                auto edgeStereo = new g2o::EdgeStereoSE3ProjectXYZ();
                edgeStereo->setId(edgeID++);
                edgeStereo->setVertex(0, landmark);
                edgeStereo->setVertex(1, frame);
                edgeStereo->fx = Camera::mfFx;
                edgeStereo->fy = Camera::mfFy;
                edgeStereo->cx = Camera::mfCx;
                edgeStereo->cy = Camera::mfCy;
                edgeStereo->bf = Camera::mfBf;
                edgeStereo->setMeasurement(g2o::Vector3((double)kp.pt.x, (double)kp.pt.y, (double)rightU));
                edgeStereo->setInformation(Eigen::Matrix3d::Identity() * pkf->getScaledFactorInv2(kp.octave));
                edgeStereo->setRobustKernel(rk);
                graph.addEdge(edgeStereo);
                edgeDB[edgeStereo] = std::make_tuple(pMp, pkf, item.second, false);
            }
        }
    }

    /// step4 进行优化
    if (isStop)
        return;

    graph.initializeOptimization();
    graph.optimize(5);
    if (!isStop) {
        for (auto &item : edgeDB) {
            auto pkf = std::get<1>(item.second);
            auto kp = pkf->getLeftKeyPoint(std::get<2>(item.second));
            auto edge = item.first;
            float sigma2 = pkf->getScaledFactor2(kp.octave);
            bool isMono = std::get<3>(item.second);
            float cov = 0;
            if (isMono) {
                auto edge = dynamic_cast<g2o::EdgeSE3ProjectXYZ *>(item.first);
                cov = 5.991;
                if (edge->chi2() > cov * sigma2)
                    edge->setLevel(1);
            } else {
                auto edge = dynamic_cast<g2o::EdgeStereoSE3ProjectXYZ *>(item.first);
                cov = 7.815;
                if (edge->chi2() > cov * sigma2)
                    edge->setLevel(1);
            }
        }
        graph.initializeOptimization();
        graph.optimize(10);
    }
    for (auto &item : edgeDB) {
        auto &pMp = std::get<0>(item.second);
        auto &pkf = std::get<1>(item.second);
        auto &idx = std::get<2>(item.second);
        auto &kp = pkf->getLeftKeyPoint(idx);
        float sigma2 = pkf->getScaledFactor2(kp.octave);
        bool isMono = std::get<3>(item.second);
        float cov = 0;
        if (isMono) {
            cov = 5.991;
            auto edge = dynamic_cast<g2o::EdgeSE3ProjectXYZ *>(item.first);
            edge->computeError();
            if (edge->chi2() > cov * sigma2 || !edge->isDepthPositive()) {
                pkf->setMapPoint(idx, nullptr);
                pMp->eraseObservetion(pkf);
            }
        } else {
            cov = 7.815;
            auto edge = dynamic_cast<g2o::EdgeStereoSE3ProjectXYZ *>(item.first);
            edge->computeError();
            if (edge->chi2() > cov * sigma2 || !edge->isDepthPositive()) {
                pkf->setMapPoint(idx, nullptr);
                pMp->eraseObservetion(pkf);
            }
        }
    }

    /// step 5设置关键帧和地图点的位姿和位置信息
    for (auto &frame : frames) {
        auto &pkf = frame.first;
        auto &vertex = frame.second;
        cv::Mat Tcw = Converter::ConvertSE32Tcw(vertex->estimate());
        if (pkf && !pkf->isBad())
            pkf->setPose(Tcw);
    }

    /// 需要先设置关键帧的位姿，再设置地图点（因为地图点依靠关键帧的数据去更新参数）
    for (auto &landmark : landmarks) {
        auto &pMp = landmark.first;
        auto &vertex = landmark.second;
        cv::Mat pos = Converter::ConvertVector32Pw(vertex->estimate());
        if (pMp && !pMp->isBad()) {
            pMp->setPos(pos);
            pMp->updateDescriptor();
            pMp->updateNormalAndDepth();
        }
    }
    KeyFrame::updateConnections(pkframe);
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

/**
 * @brief 将cv::Mat类型的Pw转换成g2o::Vector3
 *
 * @param Pw 输入的cv::Mat类型的Pw
 * @return g2o::Vector3 输出的g2o::Vector3类型的Pw
 */
g2o::Vector3 Converter::ConvertPw2Vector3(const cv::Mat &Pw) {
    float x = Pw.at<float>(0);
    float y = Pw.at<float>(1);
    float z = Pw.at<float>(2);
    return g2o::Vector3((double)x, (double)y, (double)z);
}

/**
 * @brief 将g2o::Vector3类型的Pw转换为cv::Mat类型
 *
 * @param Pw 输入的g2o::Vector3类型的Pw
 * @return cv::Mat 输出的cv::Mat类型的Pw
 */
cv::Mat Converter::ConvertVector32Pw(const g2o::Vector3 &Pw) {
    float x = Pw[0];
    float y = Pw[1];
    float z = Pw[2];
    return (cv::Mat_<float>(3, 1) << x, y, z);
}

/**
 * @brief 计算某个数据的3/4分位数
 *
 * @param data 输入的计算分位数的数据(排序好的)
 * @return double 输出的3/4分位数的值
 */
double Optimizer::ComputeThirdQuartile(const std::vector<double> &data) {
    std::size_t N = data.size();
    assert(N > 0);
    std::size_t targetId = std::floor((N - 1) * 0.75);
    double remainder = (N - 1) * 0.75 - targetId;
    if (remainder == 0) {
        return data[targetId];
    } else
        return data[targetId] * (1.0 - remainder) + remainder * (data[targetId + 1]);
}

/// Optimizer的静态变量
float Optimizer::deltaMono = std::sqrt(5.991);   ///< 单目二自由度
float Optimizer::deltaStereo = std::sqrt(7.815); ///< 双目三自由度

} // namespace ORB_SLAM2_ROS2