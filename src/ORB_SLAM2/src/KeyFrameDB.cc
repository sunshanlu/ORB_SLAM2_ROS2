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
    if (!pKf->isBowComputed())
        pKf->computeBow();
    for (const auto &item : pKf->getBowVec())
        mvConvertIdx[item.first].push_back(pKf);
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
    if (!pFrame->isBowComputed())
        pFrame->computeBow();
    std::map<KeyFrame::SharedPtr, std::size_t> kfAndWordNum; ///< 统计关键帧和相同单词数目
    for (const auto &item : pFrame->getBowVec()) {
        for (auto &kf : mvConvertIdx[item.first]) {
            if (kf && !kf->isBad()) {
                ++kfAndWordNum[kf];
            } else {
                if (kfAndWordNum.find(kf) != kfAndWordNum.end()) {
                    kfAndWordNum.erase(kf);
                }
            }
        }
    }
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

} // namespace ORB_SLAM2_ROS2
