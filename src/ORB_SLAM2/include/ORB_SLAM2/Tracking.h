#pragma once

#include "Frame.h"
#include "Map.h"

namespace ORB_SLAM2_ROS2 {
enum class TrackingState { NOT_IMAGE_YET, NOT_INITING, OK, LOST };

class Tracking {
public:
    Tracking()
        : mnFeatures(2000)
        , mnInitFeatures(4000)
        , mnMaxThresh(20)
        , mnMinThresh(7) {
        msBriefTemFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    }

    void grabFrame(cv::Mat leftImg, cv::Mat rightImg);

    /// 根据当前帧进行初始地图的初始化
    void initForStereo();

private:
    TrackingState mStatus = TrackingState::NOT_IMAGE_YET; ///< 跟踪状态

    Frame::SharedPtr mpCurrFrame; ///< 当前帧
    Frame::SharedPtr mpLastFrame; ///< 上一帧
    Map::SharedPtr mpMap;         ///< 地图

    unsigned mnFeatures;      ///< 正常跟踪时关键点数目
    unsigned mnInitFeatures;  ///< 初始化地图时关键点数目
    std::string msBriefTemFp; ///< BRIEF描述子模版路径
    int mnMaxThresh;          ///< FAST最大阈值
    int mnMinThresh;          ///< FAST最小阈值
};
} // namespace ORB_SLAM2_ROS2