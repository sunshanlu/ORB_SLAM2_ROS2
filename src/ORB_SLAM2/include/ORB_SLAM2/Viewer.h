#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace ORB_SLAM2_ROS2 {

class Map;
class Tracking;

class Viewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::weak_ptr<Tracking> TrackingWeakPtr;
    typedef std::shared_ptr<Tracking> TrackingPtr;
    typedef std::shared_ptr<Map> MapPtr;
    typedef std::shared_ptr<Viewer> SharedPtr;
    typedef Sophus::SE3f SE3;
    typedef Eigen::Vector3f Vec3;
    typedef Eigen::Matrix3f Mat3;

    Viewer(MapPtr pMap, TrackingPtr pTracker);

    void run();

    /// 绘制跟踪线程当前关键帧位姿
    void drawCurrentFrame();

    /// 绘制所有关键帧的位姿
    void drawKeyFrames();

    /// 绘制所有地图点
    void drawAllMapPoints();

    /// 绘制跟踪线程地图点
    void drawTrackingMapPoints();

    /// 设置当前帧位姿
    void setCurrFrame(const cv::Mat &Twc);

    /// 设置跟踪线程地图点
    void setTrackingMps();

    /// 设置所有关键帧位姿
    void setAllKeyFrames();

    /// 设置所有地图点
    void setAllMapPoints();

private:
    /// 绘制位姿的api
    static void drawPose(const SE3 &Twc, const std::vector<float> &vColor);

    /// 绘制地图点（局部地图点和全局地图点）
    static void drawMapPoints(const std::vector<Vec3> &vMps, const std::vector<float> &vColor);

    /// 将cv::Mat转换为Sophus::SE3
    static void convertMat2SE3(const cv::Mat &Twc, SE3 &pose);

    SE3 mCurrFrame;                     ///< 跟踪线程当前帧位姿Twc
    cv::Mat mCurrImg;                   ///< 跟踪线程当前帧图像
    std::vector<Vec3> mvTrackingMps;    ///< 跟踪线程地图点
    std::vector<SE3> mvAllKeyFrames;    ///< 所有关键帧位姿
    std::vector<Vec3> mvAllMapPoints;   ///< 全局地图点
    mutable std::mutex mMutexCurrFrame; ///< 维护mCurrFrame的互斥锁
    TrackingWeakPtr mpTracker;          ///< 跟踪线程指针
    MapPtr mpMap;                       ///< 地图指针
};

} // namespace ORB_SLAM2_ROS2
