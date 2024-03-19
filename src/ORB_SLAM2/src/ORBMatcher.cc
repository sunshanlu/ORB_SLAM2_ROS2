#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 双目相机初始化匹配
 *      1. 利用双目极线进行粗匹配
 *      2. 利用SAD+二次项差值进行精确匹配
 * @param pFrame    寻找匹配的帧
 * @return int      返回匹配的数目
 */
int ORBMatcher::searchByStereo(Frame::SharedPtr pFrame) {
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

    RowIdxDB rowIdxDB = ORBMatcher::createRowIndexDB(pFrame.get());
    for (std::size_t ldx = 0; ldx < leftKeyPoints.size(); ++ldx) {
        const auto &lKp = leftKeyPoints[ldx];
        float maxU = lKp.pt.x - 0;
        float minU = std::max(0.f, lKp.pt.x - Camera::mfFx);
        cv::Mat lDesc = leftDesc.at(ldx);
        const auto &rKpIds = rowIdxDB[cvRound(lKp.pt.y)];
        std::vector<std::size_t> candidateIdx;
        std::copy_if(rKpIds.begin(), rKpIds.end(), std::back_inserter(candidateIdx), [&](const std::size_t &idx) {
            const float &retCol = rightKeyPoints[idx].pt.x;
            return (retCol < maxU && retCol > minU) ? true : false;
        });
        if (candidateIdx.empty())
            continue;
        float ratio = 0.f;
        BestMatchDesc bestMatch = ORBMatcher::getBestMatch(lDesc, rightDesc, candidateIdx, ratio);
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
        assert(std::abs(lKp.pt.x - rightU) > 1e-2);
        pFrame->mvDepths[ldx] = Camera::mfBf / (lKp.pt.x - rightU);
        ++nMatches;
    }
    return nMatches;
}

/**
 * @brief 通过词袋加速匹配（不会丢弃恒速模型跟踪的地图点，更鲁邦）
 * @details
 *      1. 通过词袋获得pFrame和pKframe的匹配
 *      2. pFrame匹配成功的部分，跳过
 * @param pFrame    寻找匹配的普通帧
 * @param pKframe   参与匹配的关键帧
 * @return int      输出的匹配数目
 */
int ORBMatcher::searchByBow(Frame::SharedPtr pFrame, KeyFrame::SharedPtr pKframe, std::vector<cv::DMatch> &matches) {
    if (!pFrame->isBowComputed()) {
        pFrame->computeBow();
    }
    if (!pKframe->isBowComputed()) {
        pKframe->computeBow();
    }
    auto pfI = pFrame->mFeatVec.begin();
    auto pkfI = pKframe->mFeatVec.begin();
    auto pfE = pFrame->mFeatVec.end();
    auto pkfE = pKframe->mFeatVec.end();
    while (pfI != pfE && pkfI != pkfE) {
        if (pfI->first > pkfI->first)
            ++pkfI;
        else if (pfI->first < pkfI->first)
            ++pfI;
        else {
            for (const auto &pId : pfI->second) {
                const auto &pMpFrame = pFrame->mvpMapPoints[pId];
                if (pMpFrame && !pMpFrame->isBad())
                    continue;
                const auto &fDesc = pFrame->mvLeftDescriptor.at(pId);
                std::vector<std::size_t> candidateIds;
                for (const auto &pkId : pkfI->second) {
                    const auto &pMp = pKframe->mvpMapPoints[pkId];
                    if (pMp && !pMp->isBad()) {
                        candidateIds.push_back(pkId);
                    }
                }
                if (candidateIds.empty())
                    continue;
                float ratio = 0.f;
                auto bestMatch = getBestMatch(fDesc, pKframe->mvLeftDescriptor, candidateIds, ratio);
                if (bestMatch.second > mnMinThreshold || ratio > mfRatio) {
                    continue;
                }
                matches.emplace_back(pId, bestMatch.first, bestMatch.second);
            }
            ++pfI;
            ++pkfI;
        }
    }
    if (mbCheckOri) {
        verifyAngle(matches, pFrame->mvFeatsLeft, pKframe->mvFeatsLeft);
    }
    setMapPoints(pFrame->mvpMapPoints, pKframe->mvpMapPoints, matches);
    return matches.size();
}

/**
 * @brief 恒速模型跟踪中的重投影匹配方法
 * @details
 *      1. 在重投影匹配前，需要根据恒速模型确定pFrame1的位姿
 *      2. 与ORB-SLAM2不同的是，这里没有对地图点进行筛选（因为位姿并不准确，为了增加跟踪鲁棒性）
 * @param pFrame1 输入的待匹配的帧
 * @param pFrame2 输入的参与匹配的帧
 * @param th      输入的寻找匹配的阈值大小
 * @return int  输出的匹配成功的个数
 */
int ORBMatcher::searchByProjection(Frame::SharedPtr pFrame1, Frame::SharedPtr pFrame2, std::vector<cv::DMatch> &matches,
                                   float th) {
    assert(!pFrame1->mTcw.empty() && !pFrame2->mTcw.empty() && "pFrame1或pFrame的位姿为空");
    /// 判断前进或后退
    cv::Mat tlc = pFrame2->mRcw * pFrame1->mtwc + pFrame2->mtcw;
    float z = tlc.at<float>(2, 0);
    float zabs = std::abs(z);
    bool up = false, down = false;
    if (zabs > Camera::mfBl)
        z > 0 ? up = true : down = true;
    for (std::size_t idx = 0; idx < pFrame2->mvFeatsLeft.size(); ++idx) {
        MapPoint::SharedPtr pMp2 = pFrame2->mvpMapPoints[idx];
        if (!pMp2 || pMp2->isBad()) {
            continue;
        }
        std::vector<std::size_t> candidateIdx;
        const auto &feature = pFrame2->mvFeatsLeft[idx];
        const auto &desc = pFrame2->mvLeftDescriptor[idx];
        if (up) {
            candidateIdx = pFrame1->findFeaturesInArea(feature, th, feature.octave, 7);
        } else if (down) {
            candidateIdx = pFrame1->findFeaturesInArea(feature, th, 0, feature.octave);
        } else {
            int minOctave = std::max(0, feature.octave - 1);
            int maxOctave = std::min(feature.octave + 1, 7);
            candidateIdx = pFrame1->findFeaturesInArea(feature, th, minOctave, maxOctave);
        }
        if (candidateIdx.empty()) {
            continue;
        }
        std::vector<std::size_t> newCandidates;

        std::copy_if(candidateIdx.begin(), candidateIdx.end(), std::back_inserter(newCandidates),
                     [&](const std::size_t &candidateID) {
                         MapPoint::SharedPtr pMp1 = pFrame1->mvpMapPoints[candidateID];
                         if (pMp1 && !pMp1->isBad()) {
                             pMp1->addMatchInTrack();
                             return false;
                         }
                         return true;
                     });
        float ratio = 0.f;
        auto bestMatches = getBestMatch(desc, pFrame1->mvLeftDescriptor, newCandidates, ratio);
        if (ratio < mfRatio && bestMatches.second < mnMinThreshold) {
            matches.emplace_back(bestMatches.first, idx, bestMatches.second);
        }
    }
    setMapPoints(pFrame1->mvpMapPoints, pFrame2->mvpMapPoints, matches);
    return matches.size();
}

/**
 * @brief 跟踪局部地图中使用的重投影方法
 * @details
 *      1. 筛选半径需要考虑观测的方向，cos(theta) < 0.998，半径为2.5
 *      2. 筛选半径需要考虑金字塔层级，以金字塔的缩放因子作为标准差
 * @param pframe    待匹配的普通帧
 * @param mapPoints 参与匹配的局部地图中的地图点
 * @param th        最终产生的搜索半径后，需要乘的倍数
 * @return int      输出产生匹配的个数（新增的 + 已有的）
 */
int ORBMatcher::searchByProjection(Frame::SharedPtr pframe, const std::vector<MapPoint::SharedPtr> &mapPoints,
                                   float th) {
    int nMatches = 0;
    auto &pFrameMapPoints = pframe->getMapPoints();
    std::for_each(pFrameMapPoints.begin(), pFrameMapPoints.end(), [&](MapPoint::SharedPtr pMp) {
        if (pMp && !pMp->isBad())
            ++nMatches;
    });
    for (const auto &pMp : mapPoints) {
        if (!pMp || pMp->isBad())
            continue;
        float distance, cosTheta;
        cv::KeyPoint kp;
        if (!pMp->isInVision(pframe, distance, kp.pt, cosTheta))
            continue;
        kp.octave = pMp->predictLevel(distance);
        float radius = cosTheta < 0.998f ? 2.5f : 4.0f;
        int minLevel = std::max(0, kp.octave - 1);
        int maxLevel = std::min(ORBExtractor::mnLevels - 1, kp.octave + 1);
        auto candidateIdx = pframe->findFeaturesInArea(kp, radius, minLevel, maxLevel);
        if (candidateIdx.empty())
            continue;
        float fRatio;
        auto bestMatch = getBestMatch(pMp->getDesc(), pframe->getLeftDescriptor(), candidateIdx, fRatio);
        if (bestMatch.second < mnMinThreshold && fRatio < mfRatio) {
            auto &pMpInF = pframe->getMapPoints()[bestMatch.first];
            if (!pMpInF || pMpInF->isBad()) {
                pMpInF = pMp;
                ++nMatches;
            }
            pMpInF->addMatchInTrack();
        }
    }
    return nMatches;
}

/**
 * @brief 设置成功匹配的地图点
 *
 * @param toMatchMps    待匹配帧的地图点
 * @param matchMps      参与匹配的地图点
 * @param matches       匹配成功后对应的id
 */
void ORBMatcher::setMapPoints(MapPoints &toMatchMps, MapPoints &matchMps, const Matches &matches) {
    for (const auto &dmatch : matches) {
        auto &matchPMp = matchMps[dmatch.trainIdx];
        if (matchPMp && !matchPMp->isBad()) {
            matchPMp->addMatchInTrack();
            toMatchMps[dmatch.queryIdx] = matchPMp;
        } else {
            matchPMp = nullptr;
        }
    }
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
ORBMatcher::RowIdxDB ORBMatcher::createRowIndexDB(Frame *pFrame) {
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
 * @param ratio         最优比次最优的比例
 * @return ORBMatcher::BestMatchDesc pair<size_t, int>分别代表最短距离的候选索引和最短距离
 */
ORBMatcher::BestMatchDesc ORBMatcher::getBestMatch(const cv::Mat &desc, const std::vector<cv::Mat> &candidateDesc,
                                                   const std::vector<size_t> &candidateIdx, float &ratio) {
    assert(!candidateIdx.empty() && "候选描述子索引为空");
    int minDistance = INT_MAX;
    int secondDistance = INT_MAX;
    std::size_t minIdx = 0;
    for (const std::size_t &idx : candidateIdx) {
        cv::Mat cDesc = candidateDesc.at(idx);
        int distance = ORBMatcher::descDistance(desc, cDesc);
        if (distance < minDistance) {
            minDistance = distance;
            minIdx = idx;
        } else if (distance < secondDistance) {
            secondDistance = distance;
        }
    }
    ratio = (float)minDistance / (float)secondDistance;
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
    int x = cvFloor(kp.pt.x / scaleFactors[kp.octave]) + ORBExtractor::mnBorderSize + L;
    int y = cvFloor(kp.pt.y / scaleFactors[kp.octave]) + ORBExtractor::mnBorderSize;
    if (x < ORBExtractor::mnBorderSize || x > pyImg.cols - ORBExtractor::mnBorderSize - 1)
        return false;

    pitch = pyImg.rowRange(y - mnW, y + mnW + 1);
    pitch = pitch.colRange(x - mnW, x + mnW + 1);
    return true;
}

void ORBMatcher::verifyAngle(std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keyPoints1,
                             const std::vector<cv::KeyPoint> &keyPoints2) {
    HistBin hist(mnBinNum);
    for (const auto &dmatch : matches) {
        float diff = keyPoints1[dmatch.queryIdx].angle - keyPoints2[dmatch.trainIdx].angle;
        diff = diff > 0 ? diff : 360 + diff;
        int bin = diff / (360 / mnBinNum);
        hist[bin].push_back(dmatch);
    }
    std::set<std::size_t> goodBinIds;
    for (std::size_t idx = 0; idx < mnBinChoose; ++idx) {
        int maxSize = 0;
        std::size_t maxId = 0;
        bool bInit = false;
        for (std::size_t id = 0; id < mnBinNum; ++id) {
            int binSize = hist[id].size();
            if (goodBinIds.find(id) != goodBinIds.end())
                continue;
            if (binSize > maxSize) {
                maxId = id;
                maxSize = binSize;
                bInit = true;
            }
        }
        if (bInit)
            /// 代表所有的bin的大小都为0，添加与否都没意义了
            goodBinIds.insert(maxId);
    }
    std::vector<cv::DMatch> ret;
    for (auto &idx : goodBinIds)
        std::copy(hist[idx].begin(), hist[idx].end(), std::back_inserter(ret));
    std::swap(ret, matches);
}

/**
 * @brief 用于展示匹配结果
 *
 * @param image1    输入的图像1
 * @param image2    输入的图像2
 * @param keypoint1 输入的图像1的关键点
 * @param keypoint2 输入的图像2的关键点
 * @param matches   输入的两图像之间的匹配信息
 */
void ORBMatcher::showMatches(const cv::Mat &image1, const cv::Mat &image2, const std::vector<cv::KeyPoint> &keypoint1,
                             const std::vector<cv::KeyPoint> &keypoint2, const std::vector<cv::DMatch> &matches) {
    cv::Mat showImage;
    std::vector<cv::Mat> imgs{image1, image2};
    cv::hconcat(imgs, showImage);
    cv::cvtColor(showImage, showImage, cv::COLOR_GRAY2BGR);
    std::vector<cv::KeyPoint> rightKps, leftKps;
    for (const auto &dmatch : matches) {
        const auto &lkp = keypoint1[dmatch.queryIdx];
        cv::KeyPoint rkp = keypoint2[dmatch.trainIdx];
        rkp.pt.x = rkp.pt.x + image1.cols;
        cv::line(showImage, lkp.pt, rkp.pt, cv::Scalar(255, 0, 0));
        rightKps.push_back(rkp);
        leftKps.push_back(lkp);
    }
    cv::drawKeypoints(showImage, leftKps, showImage, cv::Scalar(0, 255, 0));
    cv::drawKeypoints(showImage, rightKps, showImage, cv::Scalar(0, 0, 255));
    cv::imshow("showImage", showImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int ORBMatcher::mnMaxThreshold = 100;
int ORBMatcher::mnMinThreshold = 50;
int ORBMatcher::mnMeanThreshold = 75;
int ORBMatcher::mnW = 5;
int ORBMatcher::mnL = 5;
int ORBMatcher::mnBinNum = 30;
int ORBMatcher::mnBinChoose = 3;
int ORBMatcher::mnFarParam = 40;

} // namespace ORB_SLAM2_ROS2