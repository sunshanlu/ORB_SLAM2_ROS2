#pragma once

#include <limits>

#include "ORBExtractor.h"

namespace ORB_SLAM2_ROS2 {
class Frame;
class KeyFrame;
class MapPoint;

class ORBMatcher {
public:
    typedef std::vector<std::vector<std::size_t>> RowIdxDB;
    typedef std::pair<std::size_t, int> BestMatchDesc;
    typedef std::shared_ptr<ORBMatcher> SharedPtr;
    typedef std::shared_ptr<Frame> FramePtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<MapPoint> MapPointPtr;
    typedef std::vector<std::vector<cv::DMatch>> HistBin;
    typedef std::vector<MapPointPtr> MapPoints;
    typedef std::vector<cv::DMatch> Matches;

    /// 匹配器构造，传入匹配比例和是否检查方向
    ORBMatcher(float ratio = 0.6, bool checkOri = true)
        : mfRatio(ratio)
        , mbCheckOri(checkOri) {}

    /// 利用双目图像进行搜索匹配
    int searchByStereo(FramePtr pFrame);

    /// 利用词袋加速匹配（跟踪参考关键帧，注意保留已经匹配成功的地图点）
    int searchByBow(FramePtr pFrame, KeyFramePtr pKframe, std::vector<cv::DMatch> &matches);

    /// 恒速模型中的重投影匹配
    int searchByProjection(FramePtr pFrame1, FramePtr pFrame2, std::vector<cv::DMatch> &matches, float th);

    /// 展示匹配结果
    static void showMatches(const cv::Mat &image1, const cv::Mat &image2, const std::vector<cv::KeyPoint> &keypoint1,
                            const std::vector<cv::KeyPoint> &keypoint2, const std::vector<cv::DMatch> &matches);

    /// 设置匹配成功的地图点
    void setMapPoints(MapPoints &toMatchMps, MapPoints &matchMps, const Matches &matches);

private:
    /// 精确匹配
    float pixelSADMatch(const cv::Mat &leftImage, const cv::Mat &rightImage, const cv::KeyPoint &lKp,
                        const cv::KeyPoint &rKp);

    /// 图像SAD匹配
    static float SAD(const cv::Mat &image1, const cv::Mat &image2);

    /// 创建行索引数据库
    static RowIdxDB createRowIndexDB(Frame *pFrame);

    /// 计算BRIEF描述子之间的距离（斯坦福大学的二进制统计算法）
    static int descDistance(const cv::Mat &a, const cv::Mat &b);

    /// 获取最佳匹配
    static BestMatchDesc getBestMatch(const cv::Mat &desc, const std::vector<cv::Mat> &candidateDesc,
                                      const std::vector<size_t> &candidateIdx, float &ratio);

    /// 获取图像金字塔图像中的图像块
    static bool getPitch(cv::Mat &pitch, const cv::Mat &pyImg, const cv::KeyPoint &kp, int L);

    /// 使用角度一致性验证（直方图法）
    static void verifyAngle(std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keyPoints1,
                            const std::vector<cv::KeyPoint> &keyPoints2);

public:
    static int mnMaxThreshold;  ///< BRIEF最大距离阈值
    static int mnMinThreshold;  ///< BRIEF最小距离阈值
    static int mnMeanThreshold; ///< BRIEF平均距离阈值
    static int mnW;             ///< 窗口的宽度
    static int mnL;             ///< 窗口的单侧滑动像素
    static int mnBinNum;        ///< 直方图bin的个数
    static int mnBinChoose;     ///< 选择直方图的个数
    static int mnFarParam;      ///< 明显前进或者后退参数

private:
    float mfRatio;   ///< 最佳比次最佳的比例
    bool mbCheckOri; ///< 是否进行旋转一致性验证
};

} // namespace ORB_SLAM2_ROS2
