#include <cmath>
#include <iterator>
#include <map>

#include <rclcpp/rclcpp.hpp>

#include "ORB_SLAM2/Error.h"
#include "ORB_SLAM2/ORBExtractor.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 四叉树根节点构造函数
 *
 * @param image     输入图像
 * @param keyPoints 输入的FAST角点vector
 */
QuadtreeNode::QuadtreeNode(int x, int y, const std::vector<cv::KeyPoint> &keyPoints)
    : mfRowBegin(0)
    , mfRowEnd(y)
    , mfColBegin(0)
    , mfColEnd(x)
    , mpAllKeyPoints(&keyPoints) {
    for (std::size_t i = 0; i < keyPoints.size(); ++i)
        mvKeyPoints.push_back(i);
}

/**
 * @brief 分裂节点专用构造函数
 *
 * @param pParent   父指针
 * @param rowBegin  行开始
 * @param rowEnd    行结束
 * @param colBegin  列开始
 * @param colEnd    列结束
 */
QuadtreeNode::QuadtreeNode(SharedPtr pParent, double rowBegin, double rowEnd, double colBegin, double colEnd)
    : mpParent(pParent)
    , mfRowBegin(rowBegin)
    , mfRowEnd(rowEnd)
    , mfColBegin(colBegin)
    , mfColEnd(colEnd) {
    mpAllKeyPoints = pParent->mpAllKeyPoints;
    for (auto &idx : pParent->mvKeyPoints) {
        const auto &kp = mpAllKeyPoints->at(idx);
        if (isIn(kp))
            mvKeyPoints.push_back(idx);
    }
}

/**
 * @brief 节点分裂函数
 *
 * @param pParent 指定父节点信息
 */
void QuadtreeNode::split(QuadtreeNode::SharedPtr pParent) {
    std::vector<double> rows = {mfRowBegin, (mfRowBegin + mfRowEnd) / 2, mfRowEnd};
    std::vector<double> cols = {mfColBegin, (mfColBegin + mfColEnd) / 2, mfColEnd};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            auto node = std::make_shared<QuadtreeNode>(pParent, rows[i], rows[i + 1], cols[j], cols[j + 1]);
            mvpChildren.push_back(node);
        }
    }
}

/**
 * @brief 获取节点中最大响应的特征点的索引
 *
 * @return std::size_t 最大响应特征点的索引
 */
std::size_t QuadtreeNode::getFeature() {
    std::size_t maxResIdx = 0;
    float maxResponse = 0.0;
    for (const auto &idx : mvKeyPoints) {
        auto &kp = mpAllKeyPoints->at(idx);
        if (kp.response > maxResponse) {
            maxResIdx = idx;
            maxResponse = kp.response;
        }
    }
    return std::move(maxResIdx);
}

/**
 * @brief 四叉树构造函数
 *
 * @param image     输入的图像
 * @param keypoints 输入的特征点
 * @param needNodes 需要的特征点数目
 */
Quadtree::Quadtree(int x, int y, const std::vector<cv::KeyPoint> &keypoints, unsigned int needNodes)
    : mnNeedNodes(needNodes) {
    if (keypoints.size() < mnNeedNodes) {
        throw FeatureLessError("特征点数目少于期待特征点数目，无法分裂");
    }
    mpRoot = std::make_shared<QuadtreeNode>(x, y, keypoints);
    mvNodes.insert(mpRoot);
    mnNodes = 1;
}

/**
 * @brief 四叉树分裂函数
 * @details
 *      1. 维护待分裂节点的map，节点中关键点的个数和节点指针（从大到小排序）
 *      2. 在每一次迭代过程中，只对map的第一个节点进行分裂，然后将分裂后的节点去掉
 *      3. 重复上述过程，直到节点数目大于等于需要的节点数目
 */
void Quadtree::split() {
    std::multimap<std::size_t, QuadtreeNode::SharedPtr, std::greater<std::size_t>> toSplitNodes;
    toSplitNodes.insert(std::make_pair(mpRoot->mvKeyPoints.size(), mpRoot));
    mnNodes = 1;
    // 值得注意的是，toSplitNodes这种情况不可能出现
    while (mnNodes < mnNeedNodes && !toSplitNodes.empty()) {
        auto toSplitNodeIter = toSplitNodes.begin();
        auto toSplitNode = toSplitNodeIter->second;
        toSplitNode->split(toSplitNode);
        toSplitNodes.erase(toSplitNodeIter);
        mnNodes -= 1;
        for (auto &childShared : toSplitNode->mvpChildren) {
            mvNodes.insert(childShared);
            if (childShared->mvKeyPoints.empty()) {
                continue;
            }
            toSplitNodes.insert(std::make_pair(childShared->mvKeyPoints.size(), childShared));
            mnNodes += 1;
        }
    }
    if (mnNodes < mnNeedNodes)
        throw FeatureLessError("节点数目少于期待特征点数目，分裂失败");
    nodes2kpoints(toSplitNodes);
}

/**
 * @brief 在四叉树中获取特征点
 * @details
 *      1. 获取的特征点都是节点中区域内相应最大的特征点
 *      2. 只取前mnNeedNodes个节点中的特征点
 * @param toSplitNodes 待分裂的节点，也就是经过分裂留下来的节点
 */
void Quadtree::nodes2kpoints(WeightNodes &toSplitNodes) {
    auto iter = toSplitNodes.begin();
    for (unsigned i = 0; i < mnNeedNodes; ++i) {
        auto featIdx = iter->second->getFeature();
        mvRetKpoints.insert(featIdx);
        ++iter;
    }
}

/**
 * @brief ORBExtractor的构造函数
 *
 * @param image         输入的待抽取特征点的图像
 * @param nFeatures     输入的期望抽取特征点的数目
 * @param pyramidLevels 输入的图像金字塔层级
 * @param scaleFactor   输入的图像金字塔每层缩放系数
 * @param bfTemFp       输入的BRIEF描述子的模版文件路径
 * @param maxThreshold  输入的FAST检测的最大阈值
 * @param minThreshold  输入的FAST检测的最小阈值
 */
ORBExtractor::ORBExtractor(const cv::Mat &image, int nFeatures, int pyramidLevels, float scaleFactor,
                           const std::string &bfTemFp, int maxThreshold, int minThreshold)
    : mnFeats(std::move(nFeatures))
    , mnMaxThresh(std::move(maxThreshold))
    , mnMinThresh(std::move(minThreshold)) {
    initPyramid(image, pyramidLevels, scaleFactor);
    initBriefTemplate(bfTemFp);
    initGrids(pyramidLevels);

    // 初始化最大列索引（灰度质心）
    if (!mbMaxColInit) {
        for (int idx = 0; idx < 16; ++idx)
            mvMaxColIdx.push_back(std::round(std::sqrt(mnCentroidR * mnCentroidR - idx * idx)));
        mbMaxColInit = true;
    }
}

/**
 * @brief 初始化BRIEF模版
 *
 * @param tempFp    BRIEF模版文件的路径
 */
void ORBExtractor::initBriefTemplate(const std::string &tempFp) {
    if (!mbTemInit) {
        std::ifstream ifs(tempFp);
        if (!ifs.is_open()) {
            throw FileNotOpenError("BRIEF描述子模版文件打开失败！");
        }
        std::string lineStr;
        bool isHeader = true;
        while (std::getline(ifs, lineStr)) {
            if (isHeader) {
                isHeader = false;
                continue;
            }
            std::istringstream iss(lineStr);
            cv::Point2f pt1, pt2;
            iss >> pt1.x >> pt1.y >> pt2.x >> pt2.y;
            mvBriefTem.push_back(std::make_pair(pt1, pt2));
        }
        mbTemInit = true;
    }
}

/**
 * @brief 计算金字塔
 * @details
 *      1. 维护了每层金字塔的缩放层级因子 mvfScaledFactors（static）
 *      2. 维护了每层金字塔的图像 mvPyramids
 * @param image         输入的图像信息
 * @param nLevels       输入的金字塔层数
 * @param scaleFactor   输入的金字塔的缩放因子
 */
void ORBExtractor::initPyramid(const cv::Mat &image, int nLevels, float scaleFactor) {
    mvPyramids.clear();
    mvPyramids.resize(nLevels, cv::Mat());
    if (!mbScaleInit) {
        mvnFeatures.clear();
        mvnFeatures.resize(nLevels, 0);
        for (int level = 0; level < nLevels; ++level) {
            mvfScaledFactors.push_back(std::pow(scaleFactor, level));
        }
        int sumFeatures = 0;
        int nfeats = cvRound(mnFeats * (1 - scaleFactor) / (1 - std::pow(scaleFactor, nLevels)));
        for (int i = nLevels - 1; i > 0; --i) {
            mvnFeatures[i] = nfeats;
            sumFeatures += nfeats;
            nfeats = cvRound(nfeats * scaleFactor);
        }
        mvnFeatures[0] = mnFeats - sumFeatures;
        mbScaleInit = true;
        mnLevels = nLevels;
        mfScaledFactor = scaleFactor;
    }
    // 这里首先都是拿原图做了缩放，然后添加了19的边框
    image.copyTo(mvPyramids[0]);
    cv::copyMakeBorder(image, mvPyramids[0], mnBorderSize, mnBorderSize, mnBorderSize, mnBorderSize,
                       cv::BORDER_REFLECT_101);
    int width = image.cols, height = image.rows;
    for (int i = 1; i < nLevels; ++i) {
        int widthNoBorder = cvRound(width / mvfScaledFactors[i]);
        int heightNoBorder = cvRound(height / mvfScaledFactors[i]);
        if (widthNoBorder < 10 || heightNoBorder < 10) {
            throw ImageSizeError("金字塔缩放的图像尺寸小于10像素，金字塔缩放参数需要调整");
            return;
        }
        cv::Size sz(widthNoBorder, heightNoBorder);
        cv::resize(image, mvPyramids[i], sz);
        cv::GaussianBlur(mvPyramids[i], mvPyramids[i], cv::Size(5, 5), 0);
        cv::copyMakeBorder(mvPyramids[i], mvPyramids[i], mnBorderSize, mnBorderSize, mnBorderSize, mnBorderSize,
                           cv::BORDER_REFLECT_101);
    }
}

/**
 * @brief 初始化FAST角点检测网格的范围（考虑了FAST角点检测的半径3）
 *
 * @param nLevels       输入的金字塔层级
 */
void ORBExtractor::initGrids(int nLevels) {
    if (mbIdxInit) {
        return;
    }
    mvRowIdxs.clear();
    mvColIdxs.clear();
    mvRowIdxs.resize(nLevels, std::vector<unsigned>());
    mvColIdxs.resize(nLevels, std::vector<unsigned>());
    for (int level = 0; level < nLevels; ++level) {
        cv::Mat image = mvPyramids[level];
        auto &mvRowIdx = mvRowIdxs[level];
        auto &mvColIdx = mvColIdxs[level];
        mvRowIdx.push_back(0);
        mvColIdx.push_back(0);
        unsigned row = 3, col = 3;
        while (1) {
            row += 30;
            if (row >= image.rows) {
                mvRowIdx.push_back(image.rows);
                break;
            }
            mvRowIdx.push_back(row);
        }
        while (1) {
            col += 30;
            if (col >= image.cols) {
                mvColIdx.push_back(image.cols);
                break;
            }
            mvColIdx.push_back(col);
        }
    }
    mbIdxInit = true;
}

/**
 * @brief 抽取FAST特征点，以每一块30*30的像素进行抽取
 * @details
 *     1. 将图像分为30 * 30像素的小块
 *     2. 对每一块进行FAST特征点的提取（双阈值）
 *          当特征点数目较少时，抛出‘FeatureLessError’异常
 * @param nlevel    输入的金字塔层级
 * @param keyPoints 输出的提取到的FAST角点vector
 */
void ORBExtractor::extractFast(int nlevel, std::vector<cv::KeyPoint> &keyPoints) {
    keyPoints.clear();
    std::vector<std::vector<cv::Mat>> pitches;
    const auto &image = mvPyramids[nlevel];
    getPitches(image, pitches, nlevel);
    const auto &mvRowIdx = mvRowIdxs[nlevel];
    const auto &mvColIdx = mvColIdxs[nlevel];
    const int rows = mvRowIdx.size() - 1, cols = mvColIdx.size() - 1;
    for (std::size_t idx = 0; idx < rows; ++idx) {
        for (std::size_t jdx = 0; jdx < cols; ++jdx) {
            const auto &pitch = pitches[idx][jdx];
            unsigned rowStart = mvRowIdx[idx];
            unsigned colStart = mvColIdx[jdx];
            std::vector<cv::KeyPoint> pitchKps;
            auto maxDetector = cv::FastFeatureDetector::create(mnMaxThresh, true, cv::FastFeatureDetector::TYPE_9_16);
            auto minDetector = cv::FastFeatureDetector::create(mnMinThresh, true, cv::FastFeatureDetector::TYPE_9_16);
            maxDetector->detect(pitch, pitchKps);
            if (pitchKps.empty())
                minDetector->detect(pitch, pitchKps);
            for (auto &kp : pitchKps) {
                kp.pt.x += colStart;
                kp.pt.y += rowStart;
                if (kp.pt.x < mnBorderSize || kp.pt.x > image.cols - mnBorderSize - 1 || kp.pt.y < mnBorderSize ||
                    kp.pt.y > image.rows - mnBorderSize - 1) {
                    continue;
                }
                kp.pt.x -= mnBorderSize;
                kp.pt.y -= mnBorderSize;
                kp.octave = nlevel;
                kp.pt.x *= mvfScaledFactors[nlevel];
                kp.pt.y *= mvfScaledFactors[nlevel];
                keyPoints.push_back(kp);
            }
        }
    }
    if (keyPoints.size() <= mvnFeatures[nlevel]) {
        throw FeatureLessError("图像中的特征点数目不够");
        return;
    }
}

/**
 * @brief 对给定图片进行网格划分，划分为30*30大小的网格
 *
 * @param image     输入的待划分网格的图像
 * @param pitches   输入的划分后的网格vector，以行方向的网格元素
 * @param nlevel    输入的金字塔层级
 */
void ORBExtractor::getPitches(const cv::Mat &image, std::vector<std::vector<cv::Mat>> &pitches, int nlevel) {
    pitches.clear();
    const auto &mvRowIdx = mvRowIdxs[nlevel];
    const auto &mvColIdx = mvColIdxs[nlevel];
    for (std::size_t ridx = 0; ridx < mvRowIdx.size() - 1; ++ridx) {
        std::vector<cv::Mat> rowPitches;
        for (std::size_t cidx = 0; cidx < mvColIdx.size() - 1; ++cidx) {
            rowPitches.push_back(
                image.rowRange(mvRowIdx[ridx], mvRowIdx[ridx + 1]).colRange(mvColIdx[cidx], mvColIdx[cidx + 1]));
        }
        pitches.push_back(rowPitches);
    }
}

/**
 * @brief 计算BRIEF描述子（给定特征点集合）
 * @details
 *      1. 在这个函数中，会将keyPoints进行缩小到当时的金字塔层级中去
 *      2. 在这个函数中，关键点的角度信息会被添加进来（角度不是弧度）
 * @param keyPoints     输入的关键点集合
 * @return 描述子(cv::Mat)
 */
std::vector<cv::Mat> ORBExtractor::computeBRIEF(std::vector<cv::KeyPoint> &keyPoints) {
    std::vector<cv::Mat> descriptors;
    for (int idx = 0; idx < keyPoints.size(); ++idx) {
        cv::Mat descriptorCV(1, 32, CV_8U);
        auto &keypoint = keyPoints[idx];
        auto &image = mvPyramids[keypoint.octave];
        auto &sacledFactor = mvfScaledFactors[keypoint.octave];
        cv::Point2i point(cvRound(keypoint.pt.x / sacledFactor + mnBorderSize),
                          cvRound(keypoint.pt.y / sacledFactor + mnBorderSize));
        std::vector<uchar> descriptor;
        double angle = computeBRIEF(image, point, descriptor);
        keypoint.angle = angle / M_PI * 180;
        for (int i = 0; i < 32; ++i)
            descriptorCV.at<uchar>(0, i) = descriptor[i];
        descriptors.push_back(descriptorCV);
    }
    return descriptors;
}

/**
 * @brief 计算单个特征点的BRIEF的描述子
 * @details
 *      1. 使用灰度质心法，获取关键点的灰度质心theta值
 *      2. 旋转模版点，进行BRIEF描述子的计算
 * @param image         输入的图像
 * @param point         输入的模版点
 * @param descriptor    输出的关键点的描述子（32的行向量）
 * @return double       输出的point点对应的角度信息（弧度）
 */
double ORBExtractor::computeBRIEF(const cv::Mat &image, const cv::Point2i &point, std::vector<uchar> &descriptor) {
    descriptor.clear();
    descriptor.reserve(32);
    double theta = getGrayCentroid(image, point);
    double cosVal = std::sin(theta);
    double sinVal = std::cos(theta);

    uchar value = 0;
    int bias = 0;
    for (auto &pointPair : mvBriefTem) {
        cv::Point2i p1 = std::move(rotateTemplate(pointPair.first, sinVal, cosVal));
        cv::Point2i p2 = std::move(rotateTemplate(pointPair.second, sinVal, cosVal));
        uchar p1Val = image.at<uchar>(point.y + p1.y, point.x + p1.x);
        uchar p2Val = image.at<uchar>(point.y + p2.y, point.x + p2.x);
        value |= (p1Val < p2Val) << bias;
        if (bias == 7) {
            descriptor.push_back(value);
            bias = 0;
            value = 0;
            continue;
        }
        ++bias;
    }
    return theta;
}

/**
 * @brief 计算灰度质心法
 *
 * @param image     输入的图像
 * @param point     输入的点信息
 * @return double   输出的灰度质心法计算的角度值
 */
double ORBExtractor::getGrayCentroid(const cv::Mat &image, const cv::Point2i &point) {
    auto &x = point.x;
    auto &y = point.y;
    int m10 = 0, m01 = 0;
    for (int dx = 1; dx < mnCentroidR + 1; ++dx) {
        m10 += dx * image.at<uchar>(y, x + dx);
    }
    for (int dy = 1; dy < mnCentroidR + 1; ++dy) {
        for (int dx = 0; dx < mnCentroidR + 1; ++dx) {
            uchar upValue = image.at<uchar>(y + dy, x + dx);
            uchar downValue = image.at<uchar>(y - dy, x + dx);
            m10 += dx * (upValue + downValue);
            m01 += dy * (upValue - downValue);
        }
    }
    return std::atan2((double)m01, (double)m10);
}

/**
 * @brief ORB特征点和BRIEF描述子抽取函数
 * @details
 *      1. 遍历每一层，进行ORB特征点抽取（此时的特征点会放大到原图中的位置）
 *      2. 将抽取得到的特征点都放在一块，进行四叉树均匀化特征点
 *      3. 将均匀化后的特征点进行描述子计算（在描述子计算过程中，会缩小到金字塔层级中去）
 *      4. 这种特征点均匀化后再进行描述子计算，会减少描述子的计算次数
 * @param keyPoints     输出的特征点
 * @param descriptors   输出的描述子（cv::Mat）
 */
void ORBExtractor::extract(std::vector<cv::KeyPoint> &keyPoints, std::vector<cv::Mat> &descriptors) {
    std::vector<cv::KeyPoint> fastKps;
    for (int level = 0; level < mnLevels; ++level) {
        std::vector<cv::KeyPoint> levelKps;
        extractFast(level, levelKps);
        std::copy(levelKps.begin(), levelKps.end(), std::back_inserter(fastKps));
    }
    const auto &image = mvPyramids[0];
    Quadtree quadtree(image.cols - 2 * mnBorderSize, image.rows - 2 * mnBorderSize, fastKps, mnFeats);
    quadtree.split();
    auto ids = quadtree.getFeatIdxs();
    for (const auto &id : ids)
        keyPoints.push_back(fastKps[id]);
    descriptors = computeBRIEF(keyPoints);
}

/// ORBExtractor的静态变量类外初始化
bool ORBExtractor::mbIdxInit = false;
ORBExtractor::GridIdxLevels ORBExtractor::mvRowIdxs;
ORBExtractor::GridIdxLevels ORBExtractor::mvColIdxs;
ORBExtractor::BriefTemplate ORBExtractor::mvBriefTem;
bool ORBExtractor::mbTemInit = false;
std::vector<unsigned> ORBExtractor::mvMaxColIdx;
bool ORBExtractor::mbMaxColInit = false;
int ORBExtractor::mnCentroidR = 15;
std::vector<float> ORBExtractor::mvfScaledFactors;
bool ORBExtractor::mbScaleInit = false;
int ORBExtractor::mnLevels;
std::vector<int> ORBExtractor::mvnFeatures;
int ORBExtractor::mnBorderSize = 19;
float ORBExtractor::mfScaledFactor;

/**
 * @brief BRIEF点的旋转函数
 *
 * @param toRotPoint    待旋转的点
 * @param sinValue      旋转角度sin值
 * @param cosValue      旋转角度cos值
 * @return cv::Point2i  返回旋转后的点
 */
cv::Point2i rotateTemplate(const cv::Point2f &toRotPoint, const double &sinValue, const double &cosValue) {
    cv::Point2f point;
    point.x = cvRound(toRotPoint.x * cosValue - toRotPoint.y * sinValue);
    point.y = cvRound(toRotPoint.x * sinValue + toRotPoint.y * cosValue);
    return std::move(point);
}

} // namespace ORB_SLAM2_ROS2