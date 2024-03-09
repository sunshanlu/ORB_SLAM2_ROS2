#pragma once

#include <limits>

#include "ORBExtractor.h"

namespace ORB_SLAM2_ROS2 {
class Frame;

class ORBMatcher {
public:
    typedef std::vector<std::vector<std::size_t>> RowIdxDB;
    typedef std::pair<std::size_t, int> BestMatchDesc;
    typedef std::shared_ptr<ORBMatcher> SharedPtr;

    /// 利用双目图像进行搜索匹配
    int searchByStereo(Frame* pFrame);

    ORBMatcher() = default;

private:
    /// 精确匹配
    float pixelSADMatch(const cv::Mat &leftImage, const cv::Mat &rightImage, const cv::KeyPoint &lKp,
                        const cv::KeyPoint &rKp);

    /// 图像SAD匹配
    static float SAD(const cv::Mat &image1, const cv::Mat &image2);

    /// 创建行索引数据库
    static RowIdxDB createRowIndexDB(Frame* pFrame);

    /// 计算BRIEF描述子之间的距离（斯坦福大学的二进制统计算法）
    static int descDistance(const cv::Mat &a, const cv::Mat &b);

    /// 获取最佳匹配
    static BestMatchDesc getBestMatch(cv::Mat &desc, const cv::Mat &candidateDesc,
                                      const std::vector<size_t> &candidateIdx);

    /// 获取图像金字塔图像中的图像块
    static bool getPitch(cv::Mat &pitch, const cv::Mat &pyImg, const cv::KeyPoint &kp, int L);

public:
    static int mnMaxThreshold;  /// BRIEF最大距离阈值
    static int mnMinThreshold;  /// BRIEF最小距离阈值
    static int mnMeanThreshold; /// BRIEF平均距离阈值
    static int mnW;             ///< 窗口的宽度
    static int mnL;             ///< 窗口的单侧滑动像素
};

} // namespace ORB_SLAM2_ROS2
