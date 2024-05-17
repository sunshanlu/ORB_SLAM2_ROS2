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
 * @brief 初始节点的初始分裂，这是为了防止分裂的长宽畸形问题
 * @details
 *      1. 判断宽高比，这个比值用于初始节点数目
 *      2. 根据这个节点比例，进行初始节点的初始化
 * @param pParent
 */
void QuadtreeNode::initSplit(SharedPtr pParent) {
    const double &w = mfColEnd;
    const double &h = mfRowEnd;
    const int nIni = round(w / h);
    const float hX = (double)w / nIni;
    std::vector<double> cols = {mfColBegin};
    for (std::size_t idx = 1; idx < nIni; ++idx)
        cols.push_back(idx * hX);
    cols.push_back(mfColEnd);
    for (int i = 0; i < nIni; ++i) {
        auto node = std::make_shared<QuadtreeNode>(pParent, mfRowBegin, mfRowEnd, cols[i], cols[i + 1]);
        mvpChildren.push_back(node);
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
    mpRoot->initSplit(mpRoot);
    toSplitNodes.insert(std::make_pair(mpRoot->mvKeyPoints.size(), mpRoot));
    mnNodes = 1;
    // 值得注意的是，toSplitNodes这种情况不可能出现
    while (mnNodes < mnNeedNodes && !toSplitNodes.empty()) {
        auto toSplitNodeIter = toSplitNodes.begin();
        auto toSplitNode = toSplitNodeIter->second;
        if (toSplitNode != mpRoot) {
            toSplitNode->split(toSplitNode);
        }
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
    std::size_t nodeNum = std::min((std::size_t)mnNeedNodes, toSplitNodes.size());
    for (unsigned i = 0; i < nodeNum; ++i) {
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
    initMaxU();
}

/// 初始化最大列索引（灰度质心）
void ORBExtractor::initMaxU() {
    if (!mbMaxColInit) {
        mvMaxColIdx.resize(mnCentroidR + 1);
        int v, v0, vmax = cvFloor(mnCentroidR * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(mnCentroidR * sqrt(2.f) / 2);
        const double hp2 = mnCentroidR * mnCentroidR;
        for (v = 0; v <= vmax; ++v)
            mvMaxColIdx[v] = cvRound(sqrt(hp2 - v * v));
        for (v = mnCentroidR, v0 = 0; v >= vmin; --v) {
            while (mvMaxColIdx[v0] == mvMaxColIdx[v0 + 1])
                ++v0;
            mvMaxColIdx[v] = v0;
            ++v0;
        }
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
    mvPyramids.resize(nLevels);
    mvBriefMat.resize(nLevels);

    if (!mbScaleInit) {
        mvnFeatures.resize(nLevels, 0);
        for (int level = 0; level < nLevels; ++level)
            mvfScaledFactors.push_back(std::pow(scaleFactor, level));

        float scale = 1.0f / scaleFactor;
        int sumFeatures = 0;
        int nfeats = cvRound(mnFeats * (1 - scale) / (1 - std::pow(scale, nLevels)));
        for (int level = 0; level < nLevels - 1; ++level) {
            mvnFeatures[level] = nfeats;
            sumFeatures += nfeats;
            nfeats = cvRound(nfeats * scale);
        }
        mvnFeatures[nLevels - 1] = std::max(0, mnFeats - sumFeatures);
        mbScaleInit = true;
        mnLevels = nLevels;
        mfScaledFactor = scaleFactor;
    }

    image.copyTo(mvPyramids[0]);
    int width = image.cols, height = image.rows;
    for (int i = 1; i < nLevels; ++i) {
        int levelWidth = cvRound(width / mvfScaledFactors[i]);
        int levelHeight = cvRound(height / mvfScaledFactors[i]);
        if (levelWidth < 2 * mnBorderSize || levelHeight < 2 * mnBorderSize) {
            throw ImageSizeError("金字塔缩放的图像尺寸有问题！");
            return;
        }
        cv::Size sz(levelWidth, levelHeight);
        cv::resize(image, mvPyramids[i], sz, 0, 0, cv::INTER_LINEAR);
    }
    for (int i = 0; i < nLevels; ++i)
        cv::GaussianBlur(mvPyramids[i], mvBriefMat[i], cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
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
    cv::Mat image = mvPyramids[nlevel];
    int nMaxBorderX = image.cols - mnBorderSize + 3;
    int nMaxBorderY = image.rows - mnBorderSize + 3;
    int nMinBorderX = mnBorderSize - 3;
    int nMinBorderY = mnBorderSize - 3;
    int w = nMaxBorderX - nMinBorderX;
    int h = nMaxBorderY - nMinBorderY;
    const int nCols = w / 30;
    const int nRows = h / 30;
    const int wCell = ceil(w / nCols);
    const int hCell = ceil(h / nRows);

    std::vector<cv::KeyPoint> levelKps;
    for (std::size_t idx = 0; idx < nRows; ++idx) {
        int iniY = nMinBorderY + idx * hCell;
        int maxY = iniY + hCell + 6;
        if (iniY >= nMaxBorderY - 6)
            continue;
        if (maxY > nMaxBorderY)
            maxY = nMaxBorderY;

        for (std::size_t jdx = 0; jdx < nCols; ++jdx) {
            int iniX = nMinBorderX + jdx * wCell;
            int maxX = iniX + wCell + 6;
            if (iniX >= nMaxBorderX - 6)
                continue;
            if (maxX > nMaxBorderX)
                maxX = nMaxBorderX;
            cv::Mat pitch = image.rowRange(iniY, maxY).colRange(iniX, maxX);
            std::vector<cv::KeyPoint> pitchKps;
            cv::FAST(pitch, pitchKps, mnMaxThresh, true);
            if (pitchKps.empty())
                cv::FAST(pitch, pitchKps, mnMinThresh, true);
            for (auto &kp : pitchKps) {
                kp.pt.x += jdx * wCell;
                kp.pt.y += idx * hCell;
            }
            std::copy(pitchKps.begin(), pitchKps.end(), std::back_inserter(levelKps));
        }
    }
    Quadtree quadTree(w, h, levelKps, mvnFeatures[nlevel]);
    quadTree.split();
    auto idxs = quadTree.getFeatIdxs();
    for (const auto &id : idxs) {
        auto &kp = levelKps[id];
        kp.pt.x += nMinBorderX;
        kp.pt.y += nMinBorderY;
        kp.octave = nlevel;
        keyPoints.push_back(kp);
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
        auto &sacledFactor = mvfScaledFactors[keypoint.octave];
        std::vector<uchar> descriptor;
        double angle = computeBRIEF(keypoint.octave, keypoint.pt, descriptor);
        keypoint.angle = angle / M_PI * 180;
        keypoint.pt.x *= sacledFactor;
        keypoint.pt.y *= sacledFactor;
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
double ORBExtractor::computeBRIEF(const int &nLevel, const cv::Point2f &point, std::vector<uchar> &descriptor) {
    descriptor.clear();
    descriptor.reserve(32);
    const cv::Mat &image = mvPyramids[nLevel];
    const cv::Mat &workMat = mvBriefMat[nLevel];
    double theta = getGrayCentroid(image, point);
    double cosVal = std::cos(theta);
    double sinVal = std::sin(theta);

    uchar value = 0;
    int bias = 0;
    for (auto &pointPair : mvBriefTem) {
        cv::Point2f p1 = rotateTemplate(pointPair.first, sinVal, cosVal);
        cv::Point2f p2 = rotateTemplate(pointPair.second, sinVal, cosVal);
        uchar p1Val = workMat.at<uchar>(cvRound(point.y + p1.y), cvRound(point.x + p1.x));
        uchar p2Val = workMat.at<uchar>(cvRound(point.y + p2.y), cvRound(point.x + p2.x));
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
double ORBExtractor::getGrayCentroid(const cv::Mat &image, const cv::Point2f &point) {
    int x = cvRound(point.x);
    int y = cvRound(point.y);
    int m10 = 0, m01 = 0;
    for (int dx = -mnCentroidR; dx <= mnCentroidR; ++dx)
        m10 += dx * image.at<uchar>(y, x + dx);

    for (int dy = 1; dy <= mnCentroidR; ++dy) {
        int vSum = 0;
        int d = mvMaxColIdx[dy];
        for (int dx = -d; dx <= d; ++dx) {
            uchar upValue = image.at<uchar>(y + dy, x + dx);
            uchar downValue = image.at<uchar>(y - dy, x + dx);
            m10 += dx * (upValue + downValue);
            vSum += (upValue - downValue);
        }
        m01 += vSum * dy;
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
    for (int level = 0; level < mnLevels; ++level) {
        std::vector<cv::KeyPoint> levelKps;
        extractFast(level, levelKps);
        std::copy(levelKps.begin(), levelKps.end(), std::back_inserter(keyPoints));
    }
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
 * @return cv::Point2f  返回旋转后的点
 */
cv::Point2f rotateTemplate(const cv::Point2f &toRotPoint, const double &sinValue, const double &cosValue) {
    cv::Point2f point;
    point.x = toRotPoint.x * cosValue - toRotPoint.y * sinValue;
    point.y = toRotPoint.x * sinValue + toRotPoint.y * cosValue;
    return point;
}

} // namespace ORB_SLAM2_ROS2