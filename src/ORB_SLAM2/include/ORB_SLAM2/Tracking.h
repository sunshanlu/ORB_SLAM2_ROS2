#pragma once

#include "Frame.h"

namespace ORB_SLAM2_ROS2 {
enum class TrackingState { NOT_IMAGE_YET, NOT_INITING, OK, LOST };

class Tracking {
public:
    void grabFrame(Frame::SharedPtr pFrame);

    bool initForStereo();

private:
    Frame::SharedPtr mpCurrFrame; ///< 当前帧
    Frame::SharedPtr mpRefFrame;  ///< 上一帧
};
} // namespace ORB_SLAM2_ROS2