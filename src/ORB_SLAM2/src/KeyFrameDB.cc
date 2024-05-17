#include "ORB_SLAM2/KeyFrameDB.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"

namespace ORB_SLAM2_ROS2 {

KeyFrameDB::KeyFrameDB(std::size_t nWordNum) { mvConvertIdx.resize(nWordNum); }

/**
 * @brief 向关键帧数据库中添加关键帧
 * @details
 *      1. 在添加关键帧的时候，不需考虑关键帧是否是bad（是在回环闭合中添加的关键帧）
 *      2. 以关键帧的单词ID作为索引，添加关键帧到倒排索引中去
 * @param pKf 输入的关键帧
 */
void KeyFrameDB::addKeyFrame(KeyFrame::SharedPtr pKf) {
    pKf->computeBow();
    std::unique_lock<std::mutex> lock(mMutex);
    const auto &bowVec = pKf->getBowVec();
    for (const auto &item : bowVec)
        mvConvertIdx[item.first].insert(pKf);
}

/**
 * @brief 向关键数据库中删除关键帧
 *
 * @param pKf 输入的待删除的关键帧
 */
void KeyFrameDB::eraseKeyFrame(KeyFrame::SharedPtr pKf) {
    std::unique_lock<std::mutex> lock(mMutex);
    for (auto &convertId : mvConvertIdx)
        convertId.erase(pKf);
}

/// 最小相同单词数目过滤器
void KeyFrameDB::minWordFilter(KfAndWordDB &kfAndWordNum) {
    std::size_t maxWordNum = 0;
    std::for_each(kfAndWordNum.begin(), kfAndWordNum.end(), [&](const auto &mpItem) -> void {
        if (mpItem.second > maxWordNum)
            maxWordNum = mpItem.second;
    });
    float th1 = (float)maxWordNum * 0.8;
    for (auto iter = kfAndWordNum.begin(); iter != kfAndWordNum.end();) {
        if (iter->second < th1) {
            iter = kfAndWordNum.erase(iter);
        } else
            ++iter;
    }
}

/**
 * @brief 获取关键帧和相同单词数目的数据库信息
 *
 * @param pFrame        输入的待统计的帧
 * @param kfAndWordNum  输出的统计结果
 */
void KeyFrameDB::getKfAndWordDB(VirtualFramePtr pFrame, KfAndWordDB &kfAndWordNum,
                                const std::set<KeyFramePtr> &ignoreKfs) {
    const auto &bowVec = pFrame->getBowVec();
    std::unique_lock<std::mutex> lock(mMutex);
    for (const auto &item : bowVec) {
        for (auto &kf : mvConvertIdx[item.first]) {
            if (kf && !kf->isBad()) {
                if (ignoreKfs.find(kf) != ignoreKfs.end())
                    continue;
                ++kfAndWordNum[kf];
            } else {
                if (kfAndWordNum.find(kf) != kfAndWordNum.end()) {
                    kfAndWordNum.erase(kf);
                }
            }
        }
    }
}

/**
 * @brief 寻找重定位候选关键帧
 * @details
 *      1. 统计和当前帧具有至少一个公共单词的关键帧（以0.8作为阈值，初步筛选）
 *      2. 对每个关键帧，统计是其10佳共视关键帧，同时是候选关键帧，作为候选关键帧组
 *      3. 对这个关键帧组，进行相似性总得分统计，以这个总得分作为候选关键帧组的得分
 *      4. 使用这个得分的0.75倍作为第二阈值，筛选掉一部分候选关键帧组
 *      5. 找到这些候选关键帧组中和pFrame相似程度最高的作为最后返回的候选关键帧
 * @param pFrame        输入的要进行重定位的普通帧
 * @param candidateKfs  输出的与pFrame产生重定位关系的候选关键帧
 */
void KeyFrameDB::findRelocKfs(Frame::SharedPtr pFrame, std::vector<KeyFrame::SharedPtr> &candidateKfs) {
    candidateKfs.clear();
    pFrame->computeBow();
    KfAndWordDB kfAndWordNum;

    /// 统计关键帧和相同单词数目
    getKfAndWordDB(pFrame, kfAndWordNum);

    /// 最小相同单词过滤器
    minWordFilter(kfAndWordNum);

    /// 候选关键帧组过滤器
    groupFilter(pFrame, kfAndWordNum, candidateKfs);
}

/**
 * @brief 候选关键帧组过滤器
 *
 * @param pFrame        输入的回环闭合的当前关键帧
 * @param kfAndWordNum  输入的候选关键帧和单词数目数据库
 * @param candidateKfs  输出的候选关键帧
 */
void KeyFrameDB::groupFilter(VirtualFramePtr pFrame, KfAndWordDB &kfAndWordNum,
                             std::vector<KeyFrame::SharedPtr> &candidateKfs) {
    std::vector<Group> candidateKfGroups;
    auto end = kfAndWordNum.end();
    std::size_t bestGroupIdx = 0;
    double bestAccScore = 0;
    int idx = 0;
    for (const auto &mpItem : kfAndWordNum) {
        KeyFrame::SharedPtr bestKf = mpItem.first;
        double bestScore = pFrame->computeSimilarity(*bestKf);
        Group group;
        group.mvpKfs.push_back(bestKf);
        group.mfAccScore += bestScore;
        auto kfs = mpItem.first->getOrderedConnectedKfs(10);
        for (const auto &pKf : kfs) {
            if (!pKf || pKf->isBad())
                continue;
            if (kfAndWordNum.find(pKf) != end) {
                group.mvpKfs.push_back(pKf);
                double score = pFrame->computeSimilarity(*pKf);
                group.mfAccScore += score;
                if (score > bestScore) {
                    bestScore = score;
                    bestKf = pKf;
                }
            }
        }
        group.mpBestKf = bestKf;
        candidateKfGroups.push_back(group);
        if (group.mfAccScore > bestAccScore) {
            bestAccScore = group.mfAccScore;
            bestGroupIdx = idx;
        }
        ++idx;
    }
    std::set<KeyFramePtr> candidateKfsSet;
    double th2 = bestAccScore * 0.75;
    for (const auto &group : candidateKfGroups) {
        if (group.mfAccScore > th2)
            candidateKfsSet.insert(group.mpBestKf);
    }
    std::copy(candidateKfsSet.begin(), candidateKfsSet.end(), std::back_inserter(candidateKfs));
}

/**
 * @brief 最小关键帧过滤器
 *
 * @param pFrame        输入的待寻找的回环闭合关键帧
 * @param kfAndWordNum  输入的候选关键帧和单词数目数据库
 */
void KeyFrameDB::minScoreFilter(VirtualFramePtr pFrame, KfAndWordDB &kfAndWordNum) {
    auto vConnected = pFrame->getConnectedKfs(15);
    double minScore = 1;
    if (vConnected.empty())
        minScore = 0;
    else
        for (auto &kf : vConnected) {
            if (!kf || kf->isBad())
                continue;
            double score = pFrame->computeSimilarity(*kf);
            if (score < minScore)
                minScore = score;
        }
    for (auto iter = kfAndWordNum.begin(); iter != kfAndWordNum.end();) {
        double score = pFrame->computeSimilarity(*iter->first);
        if (score < minScore)
            iter = kfAndWordNum.erase(iter);
        else
            ++iter;
    }
}

/**
 * @brief 寻找回环闭合候选关键帧
 * @details
 *      1. 最大单词数目的0.8倍作为阈值1
 *      2. 当前关键帧和它共视关键帧的最低相似度作为阈值2
 *      3. 对满足上面两个条件的关键帧，找到10个共视关键帧组成候选关键帧组
 *      4. 找到相似性得分最高的组，以其的0.75倍作为阈值3
 *
 * @param pFrame
 * @param candidateKfs
 */
void KeyFrameDB::findLoopCloseKfs(KeyFramePtr pFrame, std::vector<KeyFramePtr> &candidateKfs) {
    pFrame->computeBow();
    KfAndWordDB kfAndWordNum;

    auto vConnectedKfs = pFrame->getAllConnected();
    std::set<KeyFrame::SharedPtr> sConnectedKfs;
    for (auto &item : vConnectedKfs) {
        KeyFrame::SharedPtr pKf = item.first.lock();
        if (pKf && !pKf->isBad())
            sConnectedKfs.insert(pKf);
    }

    /// 统计关键帧和相同单词数目
    getKfAndWordDB(pFrame, kfAndWordNum, sConnectedKfs);

    /// 最小相同单词过滤器KeyFrameDB
    minWordFilter(kfAndWordNum);
    
    /// 共视关键帧过滤器
    minScoreFilter(pFrame, kfAndWordNum);

    /// 候选关键帧组过滤器
    groupFilter(pFrame, kfAndWordNum, candidateKfs);
}

} // namespace ORB_SLAM2_ROS2
