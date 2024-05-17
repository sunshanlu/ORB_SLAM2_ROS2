#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "ORB_SLAM2/Ransac.hpp"

namespace ORB_SLAM2_ROS2 {

class KeyFrame;

/// SIM3 的数据结构
struct Sim3Ret {
    Sim3Ret() = default;

    Sim3Ret(const cv::Mat &Rqp, const cv::Mat &tqp, const float &s)
        : mfS(s) {
        Rqp.copyTo(mRqp);
        tqp.copyTo(mtqp);
    }

    bool error() const { return mRqp.empty() || mtqp.empty() || mfS <= 0.0f; }

    void copyTo(Sim3Ret &other) {
        mRqp.copyTo(other.mRqp);
        mtqp.copyTo(other.mtqp);
        other.mfS = mfS;
    }

    /// 输出逆Sim3变换
    Sim3Ret inv() const {
        Sim3Ret Sinv;
        Sinv.mfS = 1.0f / mfS;
        Sinv.mRqp = mRqp.t();
        Sinv.mtqp = -Sinv.mfS * (Sinv.mRqp * mtqp);
        return Sinv;
    }

    cv::Mat mRqp; ///< Sim3旋转矩阵
    cv::Mat mtqp; ///< Sim3平移向量
    float mfS;    ///< Sim3尺度
};

/// 定义Sim3矩阵的乘法
cv::Mat operator*(const Sim3Ret &Sqp, const cv::Mat &Q3d);

/// 定义Sim3矩阵之间的乘法
Sim3Ret operator*(const Sim3Ret &Scm, const Sim3Ret &Smw);

/**
 * @brief RANSAC + SIM3
 *
 */
class Sim3Solver : public Ransac<Sim3Ret> {
public:
    typedef std::shared_ptr<Sim3Solver> SharedPtr;

    typedef std::shared_ptr<KeyFrame> KeyFramePtr;

    /// SIM3算法
    void modelFunc(const std::vector<size_t> &vIndices, Sim3Ret &modelRet) override;

    /// Sim3Solver的工厂模式
    static SharedPtr create(KeyFramePtr pKfp, KeyFramePtr pKfq, const std::vector<cv::DMatch> &pqMatches,
                            std::vector<bool> &vbChoose, bool bFixScale = true, int nMinSet = 3,
                            int nMaxIterations = 100, float fRatio = 0.4, float fProb = 0.99) {
        SharedPtr pointer(new Sim3Solver(pKfp, pKfq, pqMatches, vbChoose, true, 3, nMaxIterations, fRatio, fProb));
        return pointer;
    }

private:
    /// Sim3Solver的构造函数
    Sim3Solver(KeyFramePtr pKfp, KeyFramePtr pKfq, const std::vector<cv::DMatch> &pqMatches,
               std::vector<bool> &vbChoose, bool bFixScale = true, int nMinSet = 3, int nMaxIterations = 300,
               float fRatio = 0.4, float fProb = 0.99);

    /// 判断内外点分布
    int checkInliers(std::vector<std::size_t> &vnInlierIndices, const Sim3Ret &modelRet) override;

    /// 计算去质心坐标
    void computeRCenter(const std::vector<cv::Mat> &vP, const std::vector<cv::Mat> &vQ, cv::Mat &P, cv::Mat &Q,
                        cv::Mat &Op, cv::Mat &Oq);

    /// 重组矩阵N
    void regroupN(const cv::Mat &M, cv::Mat &N);

    /// 获取四元数转换的旋转矩阵
    void computeRotation(const cv::Mat &N, cv::Mat &Rqp);

    /// 获取s
    float computeScale(const cv::Mat &P, const cv::Mat &Q, const cv::Mat &Rqp);

    /// 计算P使用Sim3投影到Q上的2d点误差
    bool computeError(const std::size_t &idx, const Sim3Ret &Sqp);

    std::vector<cv::Mat> mvP;       ///< 匹配的P坐标3D点
    std::vector<cv::Mat> mvQ;       ///< 匹配的Q坐标3D点
    std::vector<cv::Point2f> mvP2d; ///< 匹配的P坐标对应2D点
    std::vector<cv::Point2f> mvQ2d; ///< 匹配的Q坐标对应2D点
    std::vector<float> mvfErrorsP;  ///< P坐标系下的误差
    std::vector<float> mvfErrorsQ;  ///< Q坐标系下的误差
    bool mbScaleFixed;              ///< 是否固定尺度
};
} // namespace ORB_SLAM2_ROS2