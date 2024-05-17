#pragma once

#include <fstream>
#include <memory>
#include <sstream>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_ROS2 {

class Quadtree;

/**
 * @brief 四叉树节点
 *
 */
struct QuadtreeNode {
    typedef std::shared_ptr<QuadtreeNode> SharedPtr;
    typedef std::weak_ptr<QuadtreeNode> WeakPtr;
    typedef const std::vector<cv::KeyPoint> *ConstKPsPointer;

    /// 根节点构造函数
    QuadtreeNode(int x, int y, const std::vector<cv::KeyPoint> &keyPoints);

    /// 子节点构造函数
    QuadtreeNode(SharedPtr pParent, double rowBegin, double rowEnd, double colBegin, double colEnd);

    /// 四叉树分裂函数
    void split(SharedPtr pParent);

    /// 初始节点的初始分裂
    void initSplit(SharedPtr pParent);

    /// 获取最大响应的特征点
    std::size_t getFeature();

    double mfRowBegin, mfRowEnd;          ///< 维护四叉树行索引范围
    double mfColBegin, mfColEnd;          ///< 维护四叉树列索引范围
    WeakPtr mpParent;                     ///< 父节点
    std::vector<SharedPtr> mvpChildren;   ///< 子节点
    std::vector<std::size_t> mvKeyPoints; ///< 特征点
    ConstKPsPointer mpAllKeyPoints;       ///< 所有特征点

private:
    /**
     * @brief 查看关键点是否在节点区域内
     * 这里，将节点区域上的点都放弃
     * 由于放弃了节点区域上的点，因此在FAST角点检测中，不需要极大值抑制
     * @param kp        输入的关键点
     * @return true     代表在节点内
     * @return false    代表不在节点内
     */
    bool isIn(const cv::KeyPoint &kp) {
        if (kp.pt.x > mfColBegin && kp.pt.x < mfColEnd && kp.pt.y > mfRowBegin && kp.pt.y < mfRowEnd) {
            return true;
        }
        return false;
    }
};

/**
 * @brief 四叉树
 *
 */
class Quadtree {
public:
    typedef std::shared_ptr<Quadtree> SharedPtr;
    typedef std::multimap<std::size_t, QuadtreeNode::SharedPtr, std::greater<std::size_t>> WeightNodes;

    /// 四叉树构造函数
    Quadtree(int x, int y, const std::vector<cv::KeyPoint> &keypoints, unsigned int needNodes);

    /// 四叉树管理分裂的函数
    void split();

    /// 获取四叉树均匀化的特征点id
    const std::set<std::size_t> &getFeatIdxs() { return mvRetKpoints; }

private:
    /// 节点到特征点的转换，代表四叉树的使命结束
    void nodes2kpoints(WeightNodes &toSplitNodes);

    QuadtreeNode::SharedPtr mpRoot;            ///< 四叉树根节点
    std::set<QuadtreeNode::SharedPtr> mvNodes; ///< 四叉树所有节点
    std::set<std::size_t> mvRetKpoints;        ///< 每个node中取相应最大的节点
    unsigned int mnNodes;                      ///< 节点数目
    unsigned int mnNeedNodes;                  ///< 需要的节点数目
};

/**
 * @brief 给定特征数目和图像，提取特征点和描述子
 * ORBExtractor不负责金字塔部分的实现
 */
class ORBExtractor {
public:
    typedef std::shared_ptr<ORBExtractor> SharedPtr;
    typedef std::vector<std::pair<cv::Point2f, cv::Point2f>> BriefTemplate;
    typedef std::vector<std::vector<unsigned>> GridIdxLevels;

    /// ORBExtractor的构造函数，需提供特征点数目和BRIEF模板文件路径
    ORBExtractor(const cv::Mat &image, int nFeatures, int pyramidLevels, float scaleFactor, const std::string &bfTemFp,
                 int maxThreshold, int minThreshold);

    /// 提取ORB特征点和BRIEF描述子的api
    void extract(std::vector<cv::KeyPoint> &keyPoints, std::vector<cv::Mat> &descriptors);

    /// 获取图像金字塔api
    const std::vector<cv::Mat> &getPyramid() const { return mvPyramids; }

    /// 获取金字塔缩放因子api
    static const std::vector<float> &getScaledFactors() { return mvfScaledFactors; }

private:
    /// 根据输入的图像，提取特征点并输出
    void extractFast(int nlevel, std::vector<cv::KeyPoint> &keyPoints);

    /// 利用灰度质心法，计算BRIEF描述子
    std::vector<cv::Mat> computeBRIEF(std::vector<cv::KeyPoint> &keypoints);

    /// 计算单个特征点的BRIEF描述子
    double computeBRIEF(const int &nLevel, const cv::Point2f &point, std::vector<uchar> &descriptor);

    /// 灰度质心法
    double getGrayCentroid(const cv::Mat &image, const cv::Point2f &point);

    /// 初始化图像金字塔
    void initPyramid(const cv::Mat &image, int nLevels, float scaleFactor);

    /// 初始化BRIEF模板
    void initBriefTemplate(const std::string &tempFp);

    /// 初始化maxu
    void initMaxU();

    int mnFeats;                                ///< 要抽取的特征点数目
    static bool mbIdxInit;                      ///< 划分网格的行列索引是否初始化
    static GridIdxLevels mvRowIdxs;             ///< 划分网格的行索引
    static GridIdxLevels mvColIdxs;             ///< 划分网格的列索引
    int mnMaxThresh;                            ///< FAST最大提取阈值
    int mnMinThresh;                            ///< FAST最小提取阈值
    static BriefTemplate mvBriefTem;            ///< BRIEF模板
    static bool mbTemInit;                      ///< 模版是否已经被初始化
    static std::vector<unsigned> mvMaxColIdx;   ///< 最大列索引值
    static bool mbMaxColInit;                   ///< 列索引是否初始化
    static int mnCentroidR;                     ///< 灰度质心法的半径
    std::vector<cv::Mat> mvPyramids;            ///< 图像金字塔
    std::vector<cv::Mat> mvBriefMat;            ///< 计算BRIEF描述子图像
    static std::vector<float> mvfScaledFactors; ///< 图像金字塔的缩放因子
    static bool mbScaleInit;                    ///< 金字塔缩放层级是否初始化
    static std::vector<int> mvnFeatures;        ///< 图像金字塔每层需要提取的特征点数目

public:
    static int mnLevels;         ///< 图像金字塔层数
    static int mnBorderSize;     ///< 边界宽度 (用于限制brief的边界问题)
    static float mfScaledFactor; ///< 金字塔图像层级
};

cv::Point2f rotateTemplate(const cv::Point2f &toRotPoint, const double &sinValue, const double &cosValue);

} // namespace ORB_SLAM2_ROS2