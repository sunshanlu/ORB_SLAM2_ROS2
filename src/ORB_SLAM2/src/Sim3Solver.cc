#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/Sim3Solver.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 计算SIM3变换
 * @details
 *      1. 计算去质心坐标
 *      2. 计算矩阵M
 *      3. 进行矩阵N的重组
 *      4. 对矩阵N进行特征值分解得到四元数
 * @param vIndices  输入的相对应的索引（3D点）
 * @param Rqp       输出的旋转矩阵
 * @param tqp       输出的平移向量
 * @param s         输出的尺度
 */
void Sim3Solver::modelFunc(const std::vector<size_t> &vIndices, Sim3Ret &modelRet) {
    assert(vIndices.size() >= 3 && "传入的索引不合法");
    std::vector<cv::Mat> vP, vQ;
    for (auto idx : vIndices) {
        vP.push_back(mvP[idx]);
        vQ.push_back(mvQ[idx]);
    }

    cv::Mat P, Q, Op, Oq;
    computeRCenter(vP, vQ, P, Q, Op, Oq);
    cv::Mat M = P * Q.t();
    cv::Mat N(4, 4, CV_32F);
    regroupN(M, N);
    computeRotation(N, modelRet.mRqp);
    modelRet.mfS = mbScaleFixed ? 1.0f : computeScale(P, Q, modelRet.mRqp);
    modelRet.mtqp = Oq - modelRet.mfS * modelRet.mRqp * Op;
}

/**
 * @brief 根据矩阵M重组矩阵N
 *
 * @param M 输入的矩阵M
 * @param N 输出的矩阵N
 */
void Sim3Solver::regroupN(const cv::Mat &M, cv::Mat &N) {
    const float &Sxx = M.at<float>(0, 0);
    const float &Sxy = M.at<float>(0, 1);
    const float &Sxz = M.at<float>(0, 2);
    const float &Syx = M.at<float>(1, 0);
    const float &Syy = M.at<float>(1, 1);
    const float &Syz = M.at<float>(1, 2);
    const float &Szx = M.at<float>(2, 0);
    const float &Szy = M.at<float>(2, 1);
    const float &Szz = M.at<float>(2, 2);

    N.at<float>(0, 0) = Sxx + Syy + Szz;
    N.at<float>(1, 1) = Sxx - Syy - Szz;
    N.at<float>(2, 2) = -Sxx + Syy - Szz;
    N.at<float>(3, 3) = -Sxx - Syy + Szz;
    N.at<float>(0, 1) = N.at<float>(1, 0) = Syz - Szy;
    N.at<float>(0, 2) = N.at<float>(2, 0) = Szx - Sxz;
    N.at<float>(0, 3) = N.at<float>(3, 0) = Sxy - Syx;
    N.at<float>(1, 2) = N.at<float>(2, 1) = Sxy + Syx;
    N.at<float>(1, 3) = N.at<float>(3, 1) = Szx + Sxz;
    N.at<float>(2, 3) = N.at<float>(3, 2) = Syz + Szy;
}

/**
 * @brief 计算去质心坐标
 *
 * @param vP    输入的P点
 * @param vQ    输入的Q点
 * @param P     输出的P矩阵（去质心）
 * @param Q     输出的Q矩阵（去质心）
 * @param Op    输出的P质心
 * @param Oq    输出的Q质心
 */
void Sim3Solver::computeRCenter(
    const std::vector<cv::Mat> &vP, const std::vector<cv::Mat> &vQ, cv::Mat &P, cv::Mat &Q, cv::Mat &Op, cv::Mat &Oq) {
    assert(vP.size() == vQ.size() && vP.size() >= 3 && "传入的点数量不一致");
    int nNum = vP.size();
    Op = cv::Mat::zeros(3, 1, CV_32F);
    Oq = cv::Mat::zeros(3, 1, CV_32F);
    for (std::size_t idx = 0; idx < nNum; ++idx) {
        Op += vP[idx];
        Oq += vQ[idx];
    }
    Op /= (float)nNum;
    Oq /= (float)nNum;
    std::vector<cv::Mat> vP_RCenter, vQ_RCenter;
    for (std::size_t idx = 0; idx < nNum; ++idx) {
        vP_RCenter.push_back(vP[idx] - Op);
        vQ_RCenter.push_back(vQ[idx] - Oq);
    }
    cv::hconcat(vP_RCenter, P);
    cv::hconcat(vQ_RCenter, Q);
}

/**
 * @brief 获取旋转矩阵
 * @details
 *      1. 对矩阵N进行特征值分解
 *      2. 最大特征值对应的特征向量就是四元数
 *      3. 使用cv::eigen计算对称矩阵的特征值和特征向量
 *          1) 特征值是一个4*1的向量，降序排序
 *          2) 特征向量是行排列，对应的分别是w, x, y, z
 * @param N     输入的N矩阵
 * @param Rqp   输出的旋转矩阵
 */
void Sim3Solver::computeRotation(const cv::Mat &N, cv::Mat &Rqp) {
    cv::Mat _, eigenVec;
    cv::eigen(N, _, eigenVec);
    Eigen::Quaternionf q(
        eigenVec.at<float>(0, 0), eigenVec.at<float>(0, 1), eigenVec.at<float>(0, 2), eigenVec.at<float>(0, 3));
    q.normalize();
    cv::eigen2cv(q.matrix(), Rqp);
}

/**
 * @brief 获取尺度(非对称)
 *
 * @param P     输入的P点
 * @param Q     输入的Q点
 * @return float 尺度
 *
 */
float Sim3Solver::computeScale(const cv::Mat &P, const cv::Mat &Q, const cv::Mat &Rqp) {
    float D = Q.dot(Rqp * P);
    float Sp = 0;
    for (std::size_t row = 0; row < P.rows; ++row) {
        for (std::size_t col = 0; col < P.cols; ++col) {
            const float &v = P.at<float>(row, col);
            Sp += v * v;
        }
    }
    return D / Sp;
}

/**
 * @brief SIM3 求解器的唯一构造函数
 *
 * @param pKfp              输入的p关键帧
 * @param pKfq              输入的q关键帧
 * @param pqMatches         输入的匹配点对p->q
 * @param bFixScale         输入的是否固定尺度标识
 * @param nMinSet           输入的初始化点对数量（RANSAC算法参数）
 * @param nMaxIterations    输入的的最大迭代次数（RANSAC算法参数）
 * @param fRatio            输入的内点比例（RANSAC算法参数）
 * @param fProb             输入的全是内点的概率（RANSAC算法参数）
 */
Sim3Solver::Sim3Solver(
    KeyFramePtr pKfp, KeyFramePtr pKfq, const std::vector<cv::DMatch> &pqMatches, std::vector<bool> &vbChoose,
    bool bFixScale, int nMinSet, int nMaxIterations, float fRatio, float fProb)
    : mbScaleFixed(bFixScale) {
    int idx = 0, jdx = 0;
    for (const auto &pqMatch : pqMatches) {
        const int &qIdx = pqMatch.queryIdx;
        const int &pIdx = pqMatch.trainIdx;
        auto pMpP = pKfp->getMapPoint(pIdx);
        auto pMpQ = pKfq->getMapPoint(qIdx);
        if (!pMpP || pMpP->isBad()) {
            vbChoose[jdx++] = false;
            continue;
        }
        if (!pMpQ || pMpQ->isBad()) {
            vbChoose[jdx++] = false;
            continue;
        }
        cv::Mat Rpw, tpw, Rqw, tqw;
        pKfp->getPose(Rpw, tpw);
        pKfq->getPose(Rqw, tqw);
        cv::Mat p3d = Rpw * pMpP->getPos() + tpw;
        cv::Mat q3d = Rqw * pMpQ->getPos() + tqw;
        mvP.push_back(p3d);
        mvQ.push_back(q3d);

        cv::Point2f p2d, q2d;
        Camera::project(p3d, p2d);
        Camera::project(q3d, q2d);
        mvP2d.push_back(p2d);
        mvQ2d.push_back(q2d);

        int levelP = pKfp->getLeftKeyPoint(pIdx).octave;
        int levelQ = pKfq->getLeftKeyPoint(qIdx).octave;
        mvfErrorsP.push_back(9.210 * KeyFrame::getScaledFactor2(levelP));
        mvfErrorsQ.push_back(9.210 * KeyFrame::getScaledFactor2(levelQ));
        mvAllIndices.push_back(idx++);
        ++jdx;
    }
    mnN = mvAllIndices.size();
    setRansacParams(nMinSet, nMaxIterations, fRatio, fProb);
}

/**
 * @brief 判断内点的位置
 *
 * @param vnInlierIndices   输出的判断为内点的索引位置
 * @param modelRet          输入的Sqp
 * @return int
 */
int Sim3Solver::checkInliers(std::vector<std::size_t> &vnInlierIndices, const Sim3Ret &modelRet) {
    vnInlierIndices.clear();
    int nInliers = 0;
    for (int idx = 0; idx < mnN; ++idx) {
        if (computeError(idx, modelRet)) {
            vnInlierIndices.push_back(idx);
            ++nInliers;
        }
    }
    return nInliers;
}

/**
 * @brief 计算指定索引位置的误差是否满足要求
 *
 * @param idx       输入的索引
 * @param modelRet  输入的Sqp，SIM3变换矩阵
 * @return true     误差满足要求
 * @return false    误差不满足要求
 */
bool Sim3Solver::computeError(const std::size_t &idx, const Sim3Ret &Sqp) {
    Sim3Ret Spq = Sqp.inv();
    const auto &P3d = mvP[idx];
    const auto &Q3d = mvQ[idx];
    const auto &P2d = mvP2d[idx];
    const auto &Q2d = mvQ2d[idx];
    const auto &pError = mvfErrorsP[idx];
    const auto &qError = mvfErrorsQ[idx];

    cv::Point2f P2d_, Q2d_;
    Camera::project(Sqp * P3d, Q2d_);
    Camera::project(Spq * Q3d, P2d_);

    float pError_ = std::pow(P2d_.x - P2d.x, 2) + std::pow(P2d_.y - P2d.y, 2);
    if (pError_ > pError)
        return false;
    float qError_ = std::pow(Q2d_.x - Q2d.x, 2) + std::pow(Q2d_.y - Q2d.y, 2);
    if (qError_ > qError)
        return false;
    return true;
}

/// 定义Sim3矩阵的乘法
cv::Mat operator*(const Sim3Ret &Sqp, const cv::Mat &Q3d) { return Sqp.mfS * Sqp.mRqp * Q3d + Sqp.mtqp; }

/// 定义Sim3矩阵之间的乘法
Sim3Ret operator*(const Sim3Ret &Scm, const Sim3Ret &Smw){
    float scw = Scm.mfS * Smw.mfS;
    cv::Mat Rcw = Scm.mRqp * Smw.mRqp;
    cv::Mat tcw = Scm.mfS * Scm.mRqp * Smw.mtqp + Scm.mtqp;
    return Sim3Ret(Rcw, tcw, scw);
}

} // namespace ORB_SLAM2_ROS2