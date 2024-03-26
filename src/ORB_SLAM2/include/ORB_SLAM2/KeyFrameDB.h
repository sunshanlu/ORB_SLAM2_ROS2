#pragma once

#include <memory>

#include <DBoW3/DBoW3.h>

namespace ORB_SLAM2_ROS2 {

class KeyFrame;
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
    typedef std::shared_ptr<DBoW3::Vocabulary> VocabPtr;
    typedef std::vector<std::list<KeyFramePtr>> ConvertIdx;

    KeyFrameDB(std::size_t nWordNum);

    /// 添加关键帧
    void addKeyFrame(KeyFramePtr pKf);

    /// 寻找重定位关键帧
    void findRelocKfs(FramePtr pFrame, std::vector<KeyFramePtr> &candidateKfs);

private:
    ConvertIdx mvConvertIdx; ///< 倒排索引
};

} // namespace ORB_SLAM2_ROS2