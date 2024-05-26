#pragma once
#include <thread>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "KeyFrameDB.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "Tracking.h"
#include "Viewer.h"
#include "orb_slam2_interfaces/msg/camera.hpp"
#include "orb_slam2_interfaces/msg/lost_flag.hpp"

namespace ORB_SLAM2_ROS2 {

using namespace geometry_msgs::msg;
using namespace orb_slam2_interfaces::msg;
using namespace std::placeholders;
using CameraMsg = orb_slam2_interfaces::msg::Camera;

struct Config {
    float mfSfactor;        ///< 图像金字塔缩放因子
    bool mbOnlyTracking;    ///< 系统的启动模式
    bool mbLoadMap;         ///< 系统是否加载地图
    bool mbSaveMap;         ///< 系统是否保存地图
    bool mbViewer;          ///< 是否使用可视化模块
    std::string mVocabPath; ///< 词典文件路径
    std::string mBriefPath; ///< BRIEF描述子模版
    std::string mMapPath;   ///< 地图的保存和加载路径
    float mfDScale;         ///< 深度图缩放因子
    int mDepthTh, mnFeatures, mnInitFeatures;
    int mnLevel, mninitFAST, mnminFAST;
    int mnMaxFrames, mnMinFrames, mnColorType;
    int mCameraType;
};

class System : public rclcpp::Node {
public:
    /// SLAM系统的构造函数
    System(std::string ConfigPath);

    /// 设置系统的配置
    void setSetting(const std::string &ConfigPath, Config &config);

    /// 相机帧消息回调函数
    void CameraCallback(CameraMsg::SharedPtr cameraMsg);

    /// 直接接收图像进行跟踪
    void EstimatePose(cv::Mat leftImage, cv::Mat rightImage) {
        cv::Mat pose = mpTracker->grabFrame(leftImage, rightImage);
    }

    ~System();

private:
    std::string mMapPath;                                   ///< 地图的保存和加载路径
    Map::SharedPtr mpMap;                                   ///< 地图
    KeyFrameDB::SharedPtr mpKfDB;                           ///< 关键帧数据库
    Tracking::SharedPtr mpTracker;                          ///< 跟踪对象
    LocalMapping::SharedPtr mpLocalMapper;                  ///< 局部建图对象
    LoopClosing::SharedPtr mpLoopCloser;                    ///< 回环闭合对象
    Viewer::SharedPtr mpViewer;                             ///< 可视化对象
    rclcpp::Publisher<PoseStamped>::SharedPtr mpPosePub;    ///< 位姿发布器
    rclcpp::Publisher<LostFlag>::SharedPtr mpLostPub;       ///< 跟踪丢失发布器
    rclcpp::Subscription<CameraMsg>::SharedPtr mpCameraSub; ///< 接收相机的数据
    bool mbSaveMap;                                         ///< 系统是否保存地图
    std::thread *mpLocalMapTh = nullptr;                    ///< 局部建图线程
    std::thread *mpLoopClosingTh = nullptr;                 ///< 回环闭合线程
    std::thread *mpViewerTh = nullptr;                      ///< 可视化线程
};

} // namespace ORB_SLAM2_ROS2