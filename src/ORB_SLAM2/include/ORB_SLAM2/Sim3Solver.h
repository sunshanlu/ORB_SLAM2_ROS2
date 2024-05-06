#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_ROS2 {

class Sim3Solver {
public:
    void SIM3(std::vector<size_t> vIndices, cv::Mat &Rqp, cv::Mat &tqp, float &s);

private:
    /// 计算去质心坐标
    void computeRCenter(const std::vector<cv::Mat> &vP, const std::vector<cv::Mat> &vQ, cv::Mat P, cv::Mat Q,
                        cv::Mat Op, cv::Mat Oq);

    /// 重组矩阵N
    void regroupN(const cv::Mat &M, cv::Mat &N);

    /// 获取四元数转换的旋转矩阵
    void getRotation(const cv::Mat &N, cv::Mat &Rqp);

    /// 获取s
    float getScale(const cv::Mat &P, const cv::Mat &Q, const cv::Mat &Rqp);

    std::vector<cv::Mat> mvP; ///< 匹配的P坐标3D点
    std::vector<cv::Mat> mvQ; ///< 匹配的Q坐标3D点
    bool mbScaleFixed;        ///< 是否固定尺度
    int mnMinSet;             ///< 大于等于SIM3最少匹配点数
};
} // namespace ORB_SLAM2_ROS2