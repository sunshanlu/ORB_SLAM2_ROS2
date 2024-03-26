#pragma once
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_ROS2 {

/**
 * @brief RANSAC + EPnP
 *
 */
class PnPSolver {
public:
    typedef std::shared_ptr<PnPSolver> SharedPtr;

    /// EPnP算法
    void EPnP(std::vector<std::size_t> vIndices, cv::Mat &Rcw, cv::Mat &tcw);

    /// PnPSolver的工厂模式
    static SharedPtr create(std::vector<cv::Mat> &vMapPoints, std::vector<cv::KeyPoint> &vORBPoints) {
        std::shared_ptr<PnPSolver> pointer(new PnPSolver(vMapPoints, vORBPoints));
        return pointer;
    }

    PnPSolver(const PnPSolver &other) = delete;
    PnPSolver &operator=(const PnPSolver &other) = delete;

private:
    /// PnPSolver的构造函数
    PnPSolver(std::vector<cv::Mat> &vMapPoints, std::vector<cv::KeyPoint> &vORBPoints);

    /// 获取世界坐标系下的控制点坐标
    std::vector<cv::Point3f> computeCtlPoint(const std::vector<cv::Point3f> &vMapPoints);

    /// 计算alpha矩阵
    cv::Mat computeAlpha(const std::vector<cv::Point3f> &vMapPoints, const std::vector<cv::Point3f> &vCtlPoints);

    /// 计算M矩阵
    cv::Mat computeMMat(const std::vector<cv::Point2f> &vORBPoints, const cv::Mat &alpha);

    /// 计算M矩阵的右奇异特征向量
    std::vector<std::vector<cv::Point3f>> computeMREVec(const cv::Mat &M);

    /// 计算L矩阵
    cv::Mat computeLMat(const std::vector<std::vector<cv::Point3f>> &vMREVec);

    /// 计算Rho矩阵（世界坐标系控制点的距离）
    cv::Mat computeRho(const std::vector<cv::Point3f> &vCtlPoints);

    /// 计算未经过优化的β矩阵[beta1, beta2, beta3, beta4]
    cv::Mat computeBetaUnOpt(const cv::Mat &L, const cv::Mat &rho);

    /// 获取两个地图点之间的距离平方
    static float dist2(const cv::Point3f &p1, const cv::Point3f &p2) {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    }

    /// 使用高斯牛顿法，进行beta点的精确优化
    void GLOptimize(const cv::Mat &L, cv::Mat &vbeta, const cv::Mat &rho, int maxIteration);

    /// 计算雅可比矩阵
    cv::Mat computeJacobi(const cv::Mat &L, const cv::Mat &vbeta);

    /// ICP算法计算旋转矩阵和平移向量
    void ICP(const std::vector<cv::Point3f> &ctlPointsW, const std::vector<cv::Point3f> &ctlPointsC, cv::Mat &Rcw,
             cv::Mat &tcw);

    /// 计算3*3矩阵的秩
    float computeDet3(const cv::Mat &mat);

    std::vector<cv::Point2f> mvORBPoints; ///< 2D特征点
    std::vector<cv::Point3f> mvMapPoints; ///< 3D地图点
    std::vector<float> mvfErrors;         ///< 判断内外点阈值t
};

} // namespace ORB_SLAM2_ROS2
