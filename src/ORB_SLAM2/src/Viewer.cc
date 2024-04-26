#include <gflags/gflags.h>
#include <pangolin/pangolin.h>

#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/LocalMapping.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/MapPoint.h"
#include "ORB_SLAM2/Tracking.h"
#include "ORB_SLAM2/Viewer.h"

DEFINE_double(curr_frame_r, 1.0, "当前帧，r");
DEFINE_double(curr_frame_g, 0.0, "当前帧，g");
DEFINE_double(curr_frame_b, 0.0, "当前帧，b");
DEFINE_double(key_frame_r, 0.0, "关键帧，r");
DEFINE_double(key_frame_g, 1.0, "关键帧，g");
DEFINE_double(key_frame_b, 0.0, "关键帧，b");
DEFINE_double(local_point_r, 1.0, "局部地图点，r");
DEFINE_double(local_point_g, 0.0, "局部地图点，g");
DEFINE_double(local_point_b, 0.0, "局部地图点，b");
DEFINE_double(all_point_r, 0.0, "所有地图点，r");
DEFINE_double(all_point_g, 0.0, "所有地图点，g");
DEFINE_double(all_point_b, 0.0, "所有地图点，b");
DEFINE_double(camera_line_width, 3, "相机线宽度");
DEFINE_double(keyframe_line_width, 2, "关键帧线宽度");
DEFINE_double(point_size, 2, "点大小");

namespace ORB_SLAM2_ROS2 {
using namespace std::chrono_literals;
using namespace pangolin;

Viewer::Viewer(MapPtr pMap, TrackingPtr pTracker)
    : mpMap(pMap)
    , mpTracker(pTracker) {
    mCurrFrame = SE3(Mat3::Identity(), Vec3::Zero());
}

void Viewer::run() {
    CreateWindowAndBind("ORB_SLAM2", 1080, 720);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    OpenGlRenderState sCamera(ProjectionMatrix(1080, 720, 420, 420, 540, 680, 0.1, 2000),
                              ModelViewLookAt(0, -100, 0, 0, 0, 0, AxisZ));
    Handler3D handler(sCamera);
    View &d_cam = CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0).SetHandler(&handler);
    glPointSize(2);

    while (!ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        d_cam.Activate(sCamera);

        bool updateMap = mpMap->getUpdate();
        if (updateMap) {
            mpMap->setUpdate(false);
            // setAllMapPoints();
            setAllKeyFrames();
        }
        bool updateTracker = mpTracker.lock()->getUpdate();
        if (updateTracker) {
            mpTracker.lock()->setUpdate(false);
            setTrackingMps();
        }
        // drawAllMapPoints();

        glLineWidth(FLAGS_keyframe_line_width);
        drawKeyFrames();

        glLineWidth(FLAGS_camera_line_width);
        drawCurrentFrame();
        drawTrackingMapPoints();
        FinishFrame();
        std::this_thread::sleep_for(33ms);
    }
}

/**
 * @brief 绘制位姿
 *
 * @param Twc       待绘制的位姿Twc
 * @param vColor    待绘制位姿的颜色
 */
void Viewer::drawPose(const SE3 &Twc, const std::vector<float> &vColor) {
    const static float dx = 0.8f;
    const static float dy = 0.45f;
    const static float dz = 0.7f;

    Vec3 center = Twc.translation();
    Vec3 p1 = Twc * Vec3(dx, dy, dz);
    Vec3 p2 = Twc * Vec3(dx, -dy, dz);
    Vec3 p3 = Twc * Vec3(-dx, -dy, dz);
    Vec3 p4 = Twc * Vec3(-dx, dy, dz);

    std::vector<Vec3> vPts{p1, p2, p3, p4, p1};
    glBegin(GL_LINES);
    for (std::size_t idx = 0; idx < 4; ++idx) {
        glColor3f(vColor[0], vColor[1], vColor[2]);
        glVertex3f(vPts[idx][0], vPts[idx][1], vPts[idx][2]);
        glVertex3f(vPts[idx + 1][0], vPts[idx + 1][1], vPts[idx + 1][2]);

        glColor3f(vColor[0], vColor[1], vColor[2]);
        glVertex3f(vPts[idx][0], vPts[idx][1], vPts[idx][2]);
        glVertex3f(center[0], center[1], center[2]);
    }
    glEnd();
}

/**
 * @brief 绘制当前帧
 *
 */
void Viewer::drawCurrentFrame() {
    static std::vector<float> vColor{(float)FLAGS_curr_frame_r, (float)FLAGS_curr_frame_g, (float)FLAGS_curr_frame_b};
    {
        std::unique_lock<std::mutex> lock(mMutexCurrFrame);
        drawPose(mCurrFrame, vColor);
    }
}

/**
 * @brief 绘制地图点
 *
 */
void Viewer::drawMapPoints(const std::vector<Vec3> &vMps, const std::vector<float> &vColor) {
    glBegin(GL_POINTS);
    for (const auto &p : vMps) {
        glColor4f(vColor[0], vColor[1], vColor[2], 0.5);
        glVertex3f(p[0], p[1], p[2]);
    }
    glEnd();
}

/**
 * @brief 绘制关键帧
 *
 */
void Viewer::drawKeyFrames() {
    static std::vector<float> vColor{(float)FLAGS_key_frame_r, (float)FLAGS_key_frame_g, (float)FLAGS_key_frame_b};
    for (const auto &kf : mvAllKeyFrames)
        drawPose(kf, vColor);
}

/// 绘制所有地图点
void Viewer::drawAllMapPoints() {
    static std::vector<float> vColor{(float)FLAGS_all_point_r, (float)FLAGS_all_point_g, (float)FLAGS_all_point_b};
    drawMapPoints(mvAllMapPoints, vColor);
}

/// 绘制跟踪线程地图点
void Viewer::drawTrackingMapPoints() {
    static std::vector<float> vColor{(float)FLAGS_local_point_r, (float)FLAGS_local_point_g,
                                     (float)FLAGS_local_point_b};
    drawMapPoints(mvTrackingMps, vColor);
}

/// 将cv::Mat转换为SE3类型来表示位姿
void Viewer::convertMat2SE3(const cv::Mat &Twc, SE3 &pose) {
    Mat3 Rwc;
    Vec3 twc;
    Rwc << Twc.at<float>(0, 0), Twc.at<float>(0, 1), Twc.at<float>(0, 2), Twc.at<float>(1, 0), Twc.at<float>(1, 1),
        Twc.at<float>(1, 2), Twc.at<float>(2, 0), Twc.at<float>(2, 1), Twc.at<float>(2, 2);
    twc << Twc.at<float>(0, 3), Twc.at<float>(1, 3), Twc.at<float>(2, 3);
    pose = SE3(Rwc, twc);
}

/// 设置当前帧位姿
void Viewer::setCurrFrame(const cv::Mat &Twc) {
    std::unique_lock<std::mutex> lock(mMutexCurrFrame);
    convertMat2SE3(Twc, mCurrFrame);
}

/// 设置跟踪线程地图点
void Viewer::setTrackingMps() {
    auto pMps = mpTracker.lock()->getLocalMps();
    mvTrackingMps.clear();
    for (auto &pMp : pMps) {
        if (pMp && !pMp->isBad()) {
            cv::Mat p = pMp->getPos();
            Vec3 point(p.at<float>(0), p.at<float>(1), p.at<float>(2));
            mvTrackingMps.push_back(point);
        }
    }
}

/// 设置所有关键帧位姿
void Viewer::setAllKeyFrames() {
    auto kfs = mpMap->getAllKeyFrames();
    mvAllKeyFrames.clear();
    for (const auto &kf : kfs) {
        if (!kf || kf->isBad())
            continue;
        SE3 Twc;
        convertMat2SE3(kf->getPoseInv(), Twc);
        mvAllKeyFrames.push_back(Twc);
    }
}

/// 设置所有地图点
void Viewer::setAllMapPoints() {
    auto mps = mpMap->getAllMapPoints();
    mvTrackingMps.clear();
    for (const auto &pMp : mps) {
        if (!pMp || pMp->isBad())
            continue;
        cv::Mat p = pMp->getPos();
        mvAllMapPoints.push_back(Vec3(p.at<float>(0), p.at<float>(1), p.at<float>(2)));
    }
}

} // namespace ORB_SLAM2_ROS2
