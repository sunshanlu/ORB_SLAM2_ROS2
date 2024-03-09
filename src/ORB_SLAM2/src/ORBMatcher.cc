#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Frame.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 双目相机初始化匹配
 *      1. 利用双目极线进行粗匹配
 *      2. 利用SAD+二次项差值进行精确匹配
 * @param pFrame    寻找匹配的帧
 * @return int      返回匹配的数目
 */
int ORBMatcher::searchByStereo(Frame* pFrame) {
    int nMatches = 0;
    int nLeftKp = pFrame->mvFeatsLeft.size();
    pFrame->mvDepths.clear();
    pFrame->mvFeatsRightU.clear();
    pFrame->mvDepths.resize(nLeftKp, -1.0);
    pFrame->mvFeatsRightU.resize(nLeftKp, -1.0);

    const auto &leftPyramids = pFrame->getLeftPyramid();
    const auto &rightPyramids = pFrame->getRightPyramid();
    const auto &leftKeyPoints = pFrame->getLeftKeyPoints();
    const auto &rightKeyPoints = pFrame->getRightKeyPoints();
    const auto &leftDesc = pFrame->getLeftDescriptor();
    const auto &rightDesc = pFrame->getRightDescriptor();

    RowIdxDB rowIdxDB = ORBMatcher::createRowIndexDB(pFrame);
    for (std::size_t ldx = 0; ldx < leftKeyPoints.size(); ++ldx) {
        const auto &lKp = leftKeyPoints[ldx];
        int maxU = cvRound(lKp.pt.x - 0);
        int minU = std::max(0, cvRound(lKp.pt.x - Camera::mfFx));
        cv::Mat lDesc = leftDesc.row(ldx);
        const auto &rKpIds = rowIdxDB[cvRound(lKp.pt.y)];
        std::vector<std::size_t> candidateIdx;
        std::copy_if(rKpIds.begin(), rKpIds.end(), std::back_inserter(candidateIdx), [&](const std::size_t &idx) {
            const float &retCol = rightKeyPoints[idx].pt.x;
            return (retCol < maxU && retCol > minU) ? true : false;
        });
        if (candidateIdx.empty())
            continue;
        BestMatchDesc bestMatch = ORBMatcher::getBestMatch(lDesc, rightDesc, candidateIdx);
        if (bestMatch.second > mnMeanThreshold)
            continue;

        const auto &rKp = rightKeyPoints[bestMatch.first];
        if (lKp.octave > rKp.octave + 1 || lKp.octave < rKp.octave - 1)
            continue;

        // 这里，使用之前的19像素的边界来保证窗口在滑动过程中永不越界
        const cv::Mat &leftImage = leftPyramids[lKp.octave];
        const cv::Mat &rightImage = rightPyramids[rKp.octave];
        float deltaU = pixelSADMatch(leftImage, rightImage, lKp, rKp);
        float rightU = rKp.pt.x + deltaU;
        rightU = std::max(0.f, rightU);
        rightU = std::min(rightU, (float)rightPyramids[0].cols - 1);
        pFrame->mvFeatsRightU[ldx] = rightU;
        pFrame->mvDepths[ldx] = Camera::mfBf / (lKp.pt.x - rightU);
        ++nMatches;
    }
    return nMatches;
}

/**
 * @brief 使用SAD和亚像素梯度进行精确匹配
 *
 * @param leftImage     左图（带有边界的金字塔图）
 * @param rightImage    右图（带有边界的金字塔图）
 * @param lKp           左图的特征点
 * @param rKp           右图的特征点（粗匹配成功）
 * @return float        得到的亚像素差值（精匹配失败返回0）
 */
float ORBMatcher::pixelSADMatch(const cv::Mat &leftImage, const cv::Mat &rightImage, const cv::KeyPoint &lKp,
                                const cv::KeyPoint &rKp) {
    std::vector<float> scores;
    float minScore = std::numeric_limits<float>::max();
    int bestL = 0;
    for (int l = -mnL; l < mnL + 1; ++l) {
        cv::Mat lI, rI;
        bool leftRet = getPitch(lI, leftImage, lKp, 0);
        bool rightRet = getPitch(rI, rightImage, rKp, l);
        if (!leftRet || !rightRet) {
            continue;
        }
        float score = SAD(lI, rI);
        if (score < minScore) {
            minScore = score;
            bestL = l;
        }
        scores.push_back(score);
    }
    float deltaU = 0;
    if (bestL > 0 && bestL < scores.size() - 1) {
        const float &score1 = scores[bestL - 1];
        const float &score2 = scores[bestL];
        const float &score3 = scores[bestL + 1];
        deltaU = 0.5 * (score1 - score3) / (score1 + score3 - 2 * score2);
        if (deltaU < 1 && deltaU > -1) {
            deltaU *= ORBExtractor::getScaledFactors()[rKp.octave];
        } else {
            deltaU = 0;
        }
    }
    return deltaU;
}

/**
 * @brief 根据输入的图像块，进行SAD的计算
 * @details
 *      1. 传入的图像的尺寸符合要求
 *      2. 减去图像块中心的点灰度来进行图像块初始化
 *      3. 计算得到的SAD越小，代表图像块之间越相似
 * @param image1 图像块1
 * @param image2 图像块2
 * @return float SAD值，越小代表越相似
 */
float ORBMatcher::SAD(const cv::Mat &image1, const cv::Mat &image2) {
    assert(image1.rows == 2 * mnW + 1 && image2.rows == 2 * mnW + 1 && image1.cols == 2 * mnW + 1 &&
           image2.cols == 2 * mnW + 1 && "图像不符合图像块大小的要求");
    cv::Mat i1, i2;
    image1.copyTo(i1);
    image2.copyTo(i2);
    i1.convertTo(i1, CV_32F);
    i2.convertTo(i2, CV_32F);
    auto one = cv::Mat::ones(2 * mnW + 1, 2 * mnW + 1, CV_32F);
    i1 = i1 - one * image1.at<float>(mnW, mnW);
    i2 = i2 - one * image2.at<float>(mnW, mnW);
    return cv::sum(cv::abs(i1 - i2)).val[0];
}

/**
 * @brief 构建行索引数据库
 * @details
 *      1. 以行为索引，将符合范围要求的右图特征点索引都放在一块
 *      2. 这里以固定的2px和金字塔缩放系数的乘积作为范围要求
 * @param pFrame 输入的帧
 * @return ORBMatcher::RowIdxDB vector<vector<size_t>>，行为索引，元素为符合行范围要求的右图特征点索引
 */
ORBMatcher::RowIdxDB ORBMatcher::createRowIndexDB(Frame* pFrame) {
    int rows = pFrame->getLeftImage().rows;
    int cols = pFrame->getLeftImage().cols;
    RowIdxDB rowIdxDB(rows, std::vector<std::size_t>());
    const auto &rightKps = pFrame->getRightKeyPoints();
    for (std::size_t idx = 0; idx < rightKps.size(); ++idx) {
        const auto &kp = rightKps[idx];
        float r = 2.0 * ORBExtractor::getScaledFactors()[kp.octave];
        unsigned row = cvRound(kp.pt.y);
        unsigned maxRow = std::min(rows, cvRound(row + r + 1));
        unsigned minRow = std::max(0, cvRound(row - r));
        for (unsigned row = minRow; row < maxRow; ++row)
            rowIdxDB[row].push_back(idx);
    }
    return rowIdxDB;
}

/**
 * @brief 计算描述子之间的距离（工具）
 * 斯坦福大学的二进制统计公式
 * @param a 描述子a
 * @param b 描述子b
 * @return int 描述子之间距离
 */
int ORBMatcher::descDistance(const cv::Mat &a, const cv::Mat &b) {
    assert(a.rows == 1 && b.rows == 1 && "两描述子的行数不为1");
    assert(a.cols == 32 && b.cols == 32 && "两描述子的列数不为32");
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();
    int dist = 0;
    for (int i = 0; i < 8; ++i, ++pa, ++pb) {
        unsigned v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

/**
 * @brief 从候选描述子中，找到描述子距离最短的那个
 *
 * @param desc          被匹配的单个描述子
 * @param candidateDesc 匹配图像的所有描述子
 * @param candidateIdx  匹配图像的候选描述子索引
 * @return ORBMatcher::BestMatchDesc pair<size_t, int>分别代表最短距离的候选索引和最短距离
 */
ORBMatcher::BestMatchDesc ORBMatcher::getBestMatch(cv::Mat &desc, const cv::Mat &candidateDesc,
                                                   const std::vector<size_t> &candidateIdx) {
    assert(!candidateIdx.empty() && "候选描述子索引为空");
    int minDistance = INT_MAX;
    std::size_t minIdx = 0;
    for (const std::size_t &idx : candidateIdx) {
        cv::Mat cDesc = candidateDesc.row(idx);
        int distance = ORBMatcher::descDistance(desc, cDesc);
        if (distance < minDistance) {
            minDistance = distance;
            minIdx = idx;
        }
    }
    return std::make_pair(minIdx, minDistance);
}

/**
 * @brief 获取图像块
 *
 * @param pitch 输出的图像块信息
 * @param pyImg 输入的金字塔图像（带边界）
 * @param kp    输入的特征点，提供位置
 * @param L     要在x方向上偏移的像素距离
 * @return true     图像块的中心没有超过图像边界的条件
 * @return false    图像块的中心超过了图像边界的条件
 */
bool ORBMatcher::getPitch(cv::Mat &pitch, const cv::Mat &pyImg, const cv::KeyPoint &kp, int L) {
    const auto &scaleFactors = ORBExtractor::getScaledFactors();
    int x = cvCeil(kp.pt.x / scaleFactors[kp.octave]) + ORBExtractor::mnBorderSize + L;
    int y = cvCeil(kp.pt.y / scaleFactors[kp.octave]) + ORBExtractor::mnBorderSize;
    if (x < ORBExtractor::mnBorderSize || x > pyImg.cols - ORBExtractor::mnBorderSize - 1)
        return false;

    pitch = pyImg.rowRange(y - mnW, y + mnW + 1);
    pitch = pitch.colRange(x - mnW, x + mnW + 1);
    return true;
}

int ORBMatcher::mnMaxThreshold = 100;
int ORBMatcher::mnMinThreshold = 50;
int ORBMatcher::mnMeanThreshold = 75;
int ORBMatcher::mnW = 5;
int ORBMatcher::mnL = 5;

} // namespace ORB_SLAM2_ROS2