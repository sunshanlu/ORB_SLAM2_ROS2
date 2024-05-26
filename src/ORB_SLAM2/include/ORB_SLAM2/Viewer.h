#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "Tracking.h"

namespace ORB_SLAM2_ROS2 {

class Map;
class Tracking;
class Frame;
class MapPoint;
class KeyFrame;

class Viewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::weak_ptr<Tracking> TrackingWeakPtr;
    typedef std::shared_ptr<Tracking> TrackingPtr;
    typedef std::shared_ptr<Map> MapPtr;
    typedef std::shared_ptr<Frame> FramePtr;
    typedef std::shared_ptr<Viewer> SharedPtr;
    typedef Sophus::SE3f SE3;
    typedef Eigen::Vector3f Vec3;
    typedef Eigen::Matrix3f Mat3;
    typedef std::shared_ptr<MapPoint> MapPointPtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;

    Viewer(MapPtr pMap, TrackingPtr pTracker);

    void run();

    /// 绘制跟踪线程当前关键帧位姿
    void drawCurrentFrame();

    /// 绘制所有关键帧的位姿
    void drawKeyFrames(const bool &bDrawKF, const bool &bDrawGraph);

    /// 绘制所有地图点
    void drawAllMapPoints(const bool &bDrawMP);

    /// 绘制跟踪线程地图点
    void drawTrackingMapPoints(const bool &bDrawMP);

    /// 设置当前帧位姿
    void setCurrFrame(cv::Mat trackImage, FramePtr pCurrFrame, TrackingState state);

    /// 设置跟踪线程地图点
    void setTrackingMps();

    /// 设置所有关键帧位姿
    void setAllKeyFrames();

    /// 设置所有地图点
    void setAllMapPoints();

    /// 绘制当前图像
    void drawCurrImg();

    /// 获取可视化线程是否停止
    bool isStop() const {
        std::unique_lock<std::mutex> lock(mMutexStop);
        return mbIsStop;
    }

    /// 外部线程请求停止
    void requestStop() {
        std::unique_lock<std::mutex> lock(mMutexReStop);
        mbReqestStop = true;
    }

    /// 是否请求停止
    bool isRequestStop() const {
        std::unique_lock<std::mutex> lock(mMutexReStop);
        return mbReqestStop;
    }

    /// 停止可视化线程
    void stop() {
        std::unique_lock<std::mutex> lock(mMutexStop);
        mbIsStop = true;
        cv::destroyAllWindows();
    }

private:
    /// 绘制位姿的api
    static void drawPose(const SE3 &Twc);

    /// 绘制地图点（局部地图点和全局地图点）
    static void drawMapPoints(const std::vector<MapPointPtr> &vMps);

    /// 绘制graph
    static void drawGraph(const KeyFramePtr &pKf);

    /// 将cv::Mat转换为Sophus::SE3
    static SE3 convertMat2SE3(const cv::Mat &Twc);

    SE3 mCurrPose;                           ///< 跟踪线程当前帧位姿Twc
    cv::Mat mCurrImg;                        ///< 跟踪线程当前帧图像
    FramePtr mpFrame;                        ///< 待绘制的普通帧
    std::vector<MapPointPtr> mvTrackingMps;  ///< 跟踪线程地图点
    std::vector<KeyFramePtr> mvAllKeyFrames; ///< 所有关键帧位姿
    std::vector<MapPointPtr> mvAllMapPoints; ///< 全局地图点
    mutable std::mutex mMutexCurrFrame;      ///< 维护mCurrFrame的互斥锁
    TrackingWeakPtr mpTracker;               ///< 跟踪线程指针
    MapPtr mpMap;                            ///< 地图指针
    TrackingState mState;                    ///< 跟踪线程的状态
    bool mbIsStop;                           ///< 线程是否停止
    bool mbReqestStop;                       ///< 外部线程是否请求停止
    mutable std::mutex mMutexStop;           ///< mbIsStop的互斥锁
    mutable std::mutex mMutexReStop;         ///< mbIsStop的互斥锁
    std::vector<float> mvfCurrColor;         ///< 当前帧颜色
    std::vector<float> mvfKeyColor;          ///< 关键帧颜色
    std::vector<float> mvfLPColor;           ///< 局部地图点颜色
    std::vector<float> mvfAPColor;           ///< 全局地图点颜色
    std::vector<float> mvfGHColor;           ///< 共视图颜色
    float mfCurrLW, mfKeyLW;                 ///< 线宽
    float mfPointSize;                       ///< 点的大小

public:
    static float mfdx, mfdy, mfdz; ///< 绘制金字塔形状位姿
};

} // namespace ORB_SLAM2_ROS2
