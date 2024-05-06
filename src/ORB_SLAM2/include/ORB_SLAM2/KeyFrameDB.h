#pragma once

#include <memory>
#include <mutex>
#include <set>

#include <DBoW3/DBoW3.h>

namespace ORB_SLAM2_ROS2 {

class KeyFrame;
class VirtualFrame;
class Frame;

struct Group {
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    KeyFramePtr mpBestKf;
    double mfAccScore;
    std::vector<KeyFramePtr> mvpKfs;
};

class KeyFrameDB {
public:
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<Frame> FramePtr;
    typedef std::shared_ptr<VirtualFrame> VirtualFramePtr;
    typedef std::shared_ptr<DBoW3::Vocabulary> VocabPtr;
    typedef std::vector<std::multiset<KeyFramePtr>> ConvertIdx;
    typedef std::map<KeyFrame::SharedPtr, std::size_t> KfAndWordDB;

    KeyFrameDB(std::size_t nWordNum);

    /// 添加关键帧
    void addKeyFrame(KeyFramePtr pKf);

    /// 删除关键帧
    void eraseKeyFrame(KeyFramePtr pKf);

    /// 寻找重定位关键帧
    void findRelocKfs(FramePtr pFrame, std::vector<KeyFramePtr> &candidateKfs);

    /// 寻找回环闭合候选关键帧
    void findLoopCloseKfs(KeyFramePtr pFrame, std::vector<KeyFramePtr> &candidateKfs);

private:
    /// 获取关键帧和相同单词数目的数据库信息
    void getKfAndWordDB(VirtualFramePtr pFrame, KfAndWordDB &kfAndWordNum,
                        const std::set<KeyFramePtr> &ignoreKfs = std::set<KeyFramePtr>());

    /// 最小相同单词数目过滤器
    void minWordFilter(KfAndWordDB &kfAndWordNum);

    /// 共视关键帧过滤器
    void minScoreFilter(VirtualFramePtr pFrame, KfAndWordDB &kfAndWordNum);

    /// 候选关键帧组过滤器
    void groupFilter(VirtualFramePtr pFrame, KfAndWordDB &kfAndWordNum, std::vector<KeyFramePtr> &candidateKfs);

    ConvertIdx mvConvertIdx; ///< 倒排索引
    std::mutex mMutex;       ///< 维护倒排索引的mutex
};

} // namespace ORB_SLAM2_ROS2