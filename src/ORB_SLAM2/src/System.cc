#include <cv_bridge/cv_bridge.h>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Optimizer.h"
#include "ORB_SLAM2/System.h"

namespace ORB_SLAM2_ROS2
{

using namespace std::chrono_literals;

/**
 * @brief 设置ORB_SLAM2的系统配置
 *
 * @param ConfigPath    系统的匹配文件路径
 * @param config        输出的除相机配置参数外的其他参数
 */
void System::setSetting(const std::string &ConfigPath, Config &config)
{
  cv::Mat DistCoef = cv::Mat::zeros(5, 1, CV_32F);
  cv::FileStorage setting(ConfigPath, cv::FileStorage::READ);
  if (!setting.isOpened())
  {
    RCLCPP_ERROR(get_logger(), "SLAM系统的匹配文件路径有误！");
    exit(-1);
  }
  setting["Camera.fx"] >> Camera::mfFx;
  setting["Camera.fy"] >> Camera::mfFy;
  setting["Camera.cx"] >> Camera::mfCx;
  setting["Camera.cy"] >> Camera::mfCy;
  setting["Camera.bl"] >> Camera::mfBl;
  setting["Camera.k1"] >> DistCoef.at<float>(0);
  setting["Camera.k2"] >> DistCoef.at<float>(1);
  setting["Camera.p1"] >> DistCoef.at<float>(2);
  setting["Camera.p2"] >> DistCoef.at<float>(3);
  setting["Camera.k3"] >> DistCoef.at<float>(4);
  setting["ThDepth"] >> config.mDepthTh;
  setting["ORBExtractor.nFeatures"] >> config.mnFeatures;
  setting["ORBExtractor.nInitFeatures"] >> config.mnInitFeatures;
  setting["ORBExtractor.nLevels"] >> config.mnLevel;
  setting["ORBExtractor.iniThFAST"] >> config.mninitFAST;
  setting["ORBExtractor.minThFAST"] >> config.mnminFAST;
  setting["ORBExtractor.scaleFactor"] >> config.mfSfactor;
  setting["Path.Vocabulary"] >> config.mVocabPath;
  setting["Path.BriefTemplate"] >> config.mBriefPath;
  setting["Path.Map"] >> config.mMapPath;
  setting["OnlyTracking"] >> config.mbOnlyTracking;
  setting["MaxFrames"] >> config.mnMaxFrames;
  setting["MinFrames"] >> config.mnMinFrames;
  setting["Map.LoadMap"] >> config.mbLoadMap;
  setting["Map.SaveMap"] >> config.mbSaveMap;
  setting["Viewer.UseViewer"] >> config.mbViewer;
  setting["Camera.Type"] >> config.mCameraType;
  setting["Camera.Color"] >> config.mnColorType;
  setting["Camera.DepthScale"] >> config.mfDScale;
  setting["Viewer.Posedx"] >> Viewer::mfdx;
  setting["Viewer.Posedy"] >> Viewer::mfdy;
  setting["Viewer.Posedz"] >> Viewer::mfdz;

  Camera::mfBf = Camera::mfFx * Camera::mfBl;
  Camera::mK = (cv::Mat_<float>(3, 3) << Camera::mfFx, 0, Camera::mfCx, 0, Camera::mfFy, Camera::mfCy, 0, 0, 1);
  Camera::mKInv = Camera::mK.inv();
  if (DistCoef.at<float>(4) == 0)
  {
    cv::Mat DistCoef2 = cv::Mat::zeros(4, 1, CV_32F);
    DistCoef2.at<float>(0) = DistCoef.at<float>(0);
    DistCoef2.at<float>(1) = DistCoef.at<float>(1);
    DistCoef2.at<float>(2) = DistCoef.at<float>(2);
    DistCoef2.at<float>(3) = DistCoef.at<float>(3);
    Camera::mDistCoeff = DistCoef2;
  }
  else
    Camera::mDistCoeff = DistCoef;

  if (!config.mCameraType)
    Camera::mType = CameraType::Stereo;
  else if (config.mCameraType == 1)
    Camera::mType = CameraType::RGBD;
}

System::System(std::string ConfigPath)
    : Node("ORB_SLAM2")
{
  declare_parameter("ConfigPath", "");
  std::string rosConfigPath = get_parameter("ConfigPath").as_string();
  if (!rosConfigPath.empty())
    ConfigPath = rosConfigPath;

  Config config;
  setSetting(ConfigPath, config);
  mbSaveMap = config.mbSaveMap;
  RCLCPP_INFO(get_logger(), "开始加载词袋文件，需要等待几分钟。。。。");
  auto pVocab = std::make_shared<DBoW3::Vocabulary>(config.mVocabPath);
  std::cout << std::endl;
  RCLCPP_INFO(get_logger(), "词袋文件加载成功！");
  mpMap = std::make_shared<Map>();
  mMapPath = config.mMapPath;
  if (config.mbLoadMap)
  {
    RCLCPP_INFO(get_logger(), "开始加载地图文件，需要等待几分钟。。。。");
    Map::loadFromProtobuf(config.mMapPath, mpMap);
    RCLCPP_INFO(get_logger(), "地图加载成功！");
  }
  mpKfDB = std::make_shared<KeyFrameDB>(pVocab->size());
  if (config.mbLoadMap)
  {
    auto vKfs = mpMap->getAllKeyFrames();
    for (auto &pKf : vKfs)
      mpKfDB->addKeyFrame(pKf);
  }

  mpTracker = std::make_shared<Tracking>(mpMap, mpKfDB, pVocab, config.mBriefPath, config.mnFeatures, config.mnInitFeatures, config.mninitFAST,
                                         config.mnminFAST, config.mDepthTh, config.mnMaxFrames, config.mnMinFrames, config.mnColorType, config.mbOnlyTracking,
                                         config.mfDScale, config.mnLevel, config.mfSfactor);
  if (config.mbViewer)
  {
    mpViewer = std::make_shared<Viewer>(mpMap, mpTracker);
    mpTracker->setViewer(mpViewer);
    mpViewerTh = new std::thread(&Viewer::run, mpViewer);
  }

  if (!config.mbOnlyTracking)
  {
    mpLocalMapper = std::make_shared<LocalMapping>(mpMap);
    mpLoopCloser = std::make_shared<LoopClosing>(mpKfDB, mpMap, mpLocalMapper, mpTracker);
    mpTracker->setLocalMapper(mpLocalMapper);
    mpLocalMapper->setLoopClosing(mpLoopCloser);
    mpLocalMapTh = new std::thread(&LocalMapping::run, mpLocalMapper);
    mpLoopClosingTh = new std::thread(&LoopClosing::run, mpLoopCloser);
  }

  mpPosePub = create_publisher<PoseStamped>("ORB_SLAM2/Pose", 10);
  mpLostPub = create_publisher<LostFlag>("ORB_SLAM2/Lost", 10);
  mpCameraSub = create_subscription<CameraMsg>("ORB_SLAM2/Camera", 10, std::bind(&System::CameraCallback, this, _1));
}

/**
 * @brief 相机数目的回调函数
 *
 * @param cameraMsg 输入的相机数据指针
 */
void System::CameraCallback(CameraMsg::SharedPtr cameraMsg)
{
  auto lImagePtr = cv_bridge::toCvCopy(cameraMsg->image0, cameraMsg->image0.encoding);
  auto rImagePtr = cv_bridge::toCvCopy(cameraMsg->image1, cameraMsg->image1.encoding);
  cv::Mat pose = mpTracker->grabFrame(lImagePtr->image, rImagePtr->image);
  if (pose.empty())
  {
    LostFlag msg;
    msg.header.frame_id = "Camera";
    msg.header.stamp = now();
    msg.lost.data = true;
    mpLostPub->publish(msg);
    return;
  }
  auto Qcw = Converter::ConvertCV2Eigen(pose.rowRange(0, 3).colRange(0, 3));
  PoseStamped msg;
  msg.header.frame_id = "Camera";
  msg.header.stamp = now();
  msg.pose.position.x = pose.at<float>(0, 3);
  msg.pose.position.y = pose.at<float>(1, 3);
  msg.pose.position.z = pose.at<float>(2, 3);
  msg.pose.orientation.x = Qcw.x();
  msg.pose.orientation.y = Qcw.y();
  msg.pose.orientation.z = Qcw.z();
  msg.pose.orientation.w = Qcw.w();
  mpPosePub->publish(msg);
}

System::~System()
{
  if (mpLocalMapper)
  {
    mpLocalMapper->setFinished();
    mpLocalMapTh->join();
    delete mpLocalMapTh;
  }
  if (mpLoopCloser)
  {
    mpLoopCloser->requestStop();
    while (!mpLoopCloser->isStop())
      std::this_thread::sleep_for(1ms);
    mpLoopClosingTh->join();
    delete mpLoopClosingTh;
  }
  if (mpViewer)
  {
    mpViewer->requestStop();
    while (!mpViewer->isStop())
      std::this_thread::sleep_for(1ms);
    mpViewerTh->join();
    delete mpViewerTh;
  }
  if (mbSaveMap)
  {
    RCLCPP_INFO(rclcpp::get_logger("ORB_SLAM2"), "正在保存地图，请稍后");
    mpMap->saveToProtobuf(mMapPath);
  }
}

} // namespace ORB_SLAM2_ROS2