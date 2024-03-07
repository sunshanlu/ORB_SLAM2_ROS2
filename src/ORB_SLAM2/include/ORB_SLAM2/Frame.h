#pragma once
#include <memory>
#include <vector>

#include "MapPoint.h"

namespace ORB_SLAM2_ROS2 {
class MapPint;
class ORBFeature;

class Frame {

public:
    typedef std::shared_ptr<Frame> SharedPtr;

private:
    int mnID;                                     ///< 帧ID
    std::vector<ORBFeature> mvFeatsLeft;          ///< 左图特征点坐标
    std::vector<double> mvDepths;                 ///< 特征点对应的深度值
    std::vector<double> mvFeatsRight;             ///< 右图特征点坐标
    std::vector<MapPoint::SharedPtr> mvMapPoints; ///< 左图对应的地图点
};
} // namespace ORB_SLAM2_ROS2