#include "ORB_SLAM2/LocalClosing.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/KeyFrameDB.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 判断是否有满足回环闭合关键帧存在
 * @details
 *      1. 统计候选组与连续组之间的连续性关系
 *          1) 如果发生连续，更新当前的连续组链的内容和长度
 *          2) 注意，发生过连续性条件的连续组不会重复进行检测
 *      2. 检测连续性条件，即检测当前连续组链中，是否有满足连续性条件的连续组
 *          1) 如果有满足连续性条件的连续组，将连续组对应的候选关键帧加入到mvEnoughKfs中，返回true
 *          2) 如果没有满足连续性条件的连续组，返回false
 * @return true     有满足连续性条件的回环闭合关键帧
 * @return false    没有满足连续性条件的回环闭合关键帧
 */
bool LocalMapping::detectLoop() {
    mvEnoughKfs.clear();
    std::vector<KeyFrame::SharedPtr> vpLoopCandidates;
    mpKeyFrameDB->findLoopCloseKfs(mpCurrKeyFrame, vpLoopCandidates);
    if (vpLoopCandidates.empty())
        return false;
    decltype(mvConsistGroups) vCurrConsistGroups;

    bool groupEmpty = mvConsistGroups.empty();
    std::vector<bool> vbConsist(mvConsistGroups.size(), false);
    for (auto &pkf : vpLoopCandidates) {
        auto mCandidateConnecteds = pkf->getAllConnected();
        std::set<KeyFrame::SharedPtr> sCandidateGroup;
        for (auto &item : mCandidateConnecteds) {
            auto pKf = item.first.lock();
            if (pkf && !pkf->isBad())
                sCandidateGroup.insert(pKf);
        }
        sCandidateGroup.insert(pkf);
        if (groupEmpty) {
            mvConsistGroups.push_back(std::make_pair(sCandidateGroup, 0));
            continue;
        }

        bool isConsist = false;
        for (std::size_t idx = 0; idx < mvConsistGroups.size(); ++idx) {
            if (vbConsist[idx])
                continue;
            auto &sConsistGroup = mvConsistGroups[idx].first;
            int currLen = mvConsistGroups[idx].second;
            for (auto &pkf : sCandidateGroup) {
                if (!pkf || pkf->isBad())
                    continue;
                auto iter = sConsistGroup.find(pkf);
                if (iter != sConsistGroup.end()) {
                    currLen = currLen += 1;
                    isConsist = true;
                    break;
                }
            }
            if (isConsist) {
                vCurrConsistGroups.push_back(std::make_pair(sCandidateGroup, currLen));
                vbConsist[idx] = true;
                break;
            }
        }
        if (!isConsist)
            vCurrConsistGroups.push_back(std::make_pair(sCandidateGroup, 0));
    }

    assert(vCurrConsistGroups.size() == vpLoopCandidates.size() && "候选关键帧数目和当前连续组的长度应该一致");
    for (std::size_t idx = 0; idx < vCurrConsistGroups.size(); ++idx) {
        auto &currConsistGroup = vCurrConsistGroups[idx];
        auto &candidateKf = vpLoopCandidates[idx];
        if (currConsistGroup.second > 3) {
            mvEnoughKfs.push_back(candidateKf);
        }
    }
    std::swap(mvConsistGroups, vCurrConsistGroups);
    return !mvEnoughKfs.empty();
}

} // namespace ORB_SLAM2_ROS2