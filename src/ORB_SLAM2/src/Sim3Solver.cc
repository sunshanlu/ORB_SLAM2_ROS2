#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

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
void Sim3Solver::SIM3(std::vector<size_t> vIndices, cv::Mat &Rqp, cv::Mat &tqp, float &s) {
    assert(vIndices.size() >= 3 && "传入的索引不合法");
    std::vector<cv::Mat> vP, vQ;
    cv::Mat P, Q, Op, Oq;
    computeRCenter(vP, vQ, P, Q, Op, Oq);
    cv::Mat M = P * Q.t();
    cv::Mat N(4, 4, CV_32F);
    regroupN(M, N);
    getRotation(N, Rqp);
    s = mbScaleFixed ? 1.0f : getScale(P, Q, Rqp);
    tqp = Oq - s * Rqp * Op;
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
void Sim3Solver::computeRCenter(const std::vector<cv::Mat> &vP, const std::vector<cv::Mat> &vQ, cv::Mat P, cv::Mat Q,
                                cv::Mat Op, cv::Mat Oq) {
    assert(vP.size() == vQ.size() && vP.size() >= 3 && "传入的点数量不一致");
    int nNum = vP.size();
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
void Sim3Solver::getRotation(const cv::Mat &N, cv::Mat &Rqp) {
    cv::Mat _, eigenVec;
    cv::eigen(N, _, eigenVec);
    Eigen::Quaternionf q(eigenVec.at<float>(0, 0), eigenVec.at<float>(0, 1), eigenVec.at<float>(0, 2),
                         eigenVec.at<float>(0, 3));
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
float Sim3Solver::getScale(const cv::Mat &P, const cv::Mat &Q, const cv::Mat &Rqp) {
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

} // namespace ORB_SLAM2_ROS2